#!/usr/bin/env python3
"""
File store utility - put/get files to/from remote folder.
Based on Google API, used by pack.sh and unpack.sh.

Usage:
    python3 store.py put   <file>  [--folder NAME]
    python3 store.py get   <name>  [--folder NAME] [--dest DIR]
    python3 store.py latest         [--folder NAME] [--dest DIR]
    python3 store.py ls             [--folder NAME]
    python3 store.py setup
"""

import os
import sys
import argparse
import time
import ssl
import json

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(SCRIPT_DIR, 'token.json')
CRED_PATH = os.path.join(SCRIPT_DIR, 'credentials.json')
DEFAULT_FOLDER = 'packs'
RETRY_COUNT = 4
RETRY_DELAY = 1.0
API_BASE = 'https://www.googleapis.com/drive/v3'
UPLOAD_BASE = 'https://www.googleapis.com/upload/drive/v3'
CHUNK_SIZE = 1024 * 1024


class RetryableHttpError(Exception):
    def __init__(self, response):
        body = response.text.strip().replace('\n', ' ')[:200]
        super().__init__(f"HTTP {response.status_code}: {body}")
        self.response = response


def _is_retryable_exception(exc):
    if isinstance(exc, (ssl.SSLError, OSError, RetryableHttpError)):
        return True
    try:
        import requests
        return isinstance(exc, requests.exceptions.RequestException)
    except Exception:
        return False


def _retry(label, fn, retries=RETRY_COUNT, delay=RETRY_DELAY):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable_exception(exc):
                raise
            last_exc = exc
            if attempt == retries:
                raise
            wait = delay * attempt
            print(
                f"{label} failed ({exc.__class__.__name__}: {exc}); retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise last_exc


def _q(value):
    return value.replace('\\', '\\\\').replace("'", "\\'")


def _request(label, session, method, url, **kwargs):
    kwargs.setdefault('timeout', 60)

    def send():
        resp = session.request(method, url, **kwargs)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableHttpError(resp)
        if resp.status_code >= 400:
            body = resp.text.strip()[:500]
            raise RuntimeError(f"{label} failed with HTTP {resp.status_code}: {body}")
        return resp

    return _retry(label, send)


def _auth():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            _retry('refresh token', lambda: creds.refresh(Request()))
        else:
            if not os.path.exists(CRED_PATH):
                print(f"error: {CRED_PATH} not found")
                print("run: python3 store.py setup")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    return creds


def _service():
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(_auth())


def _find_folder(service, name, parent_id=None):
    q = f"mimeType='application/vnd.google-apps.folder' and name='{_q(name)}' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    r = _request(
        f"list folder {name}",
        service,
        'GET',
        f'{API_BASE}/files',
        params={'q': q, 'spaces': 'drive', 'fields': 'files(id,name)'},
    ).json()
    items = r.get('files', [])
    if items:
        return items[0]['id']
    meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        meta['parents'] = [parent_id]
    f = _request(
        f"create folder {name}",
        service,
        'POST',
        f'{API_BASE}/files',
        params={'fields': 'id'},
        json=meta,
    ).json()
    print(f"created folder: {name}")
    return f['id']


def cmd_put(args):
    if not os.path.isfile(args.file):
        print(f"error: {args.file} not found")
        sys.exit(1)

    svc = _service()
    folder_id = _find_folder(svc, args.folder)
    fname = os.path.basename(args.file)
    fsize = os.path.getsize(args.file)

    meta = {'name': fname, 'parents': [folder_id]}
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'X-Upload-Content-Type': 'application/octet-stream',
        'X-Upload-Content-Length': str(fsize),
    }
    init_resp = _request(
        f"start upload {fname}",
        svc,
        'POST',
        f'{UPLOAD_BASE}/files',
        params={'uploadType': 'resumable', 'fields': 'id,name'},
        headers=headers,
        data=json.dumps(meta),
    )
    upload_url = init_resp.headers.get('Location')
    if not upload_url:
        raise RuntimeError('Drive did not return an upload URL')

    print(f"putting {fname} ({fsize / 1024 / 1024:.1f} MB)...")
    t0 = time.time()

    def upload():
        with open(args.file, 'rb') as fh:
            upload_resp = svc.request(
                'PUT',
                upload_url,
                headers={
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(fsize),
                },
                data=fh,
                timeout=(30, 600),
            )
        if upload_resp.status_code in (429, 500, 502, 503, 504):
            raise RetryableHttpError(upload_resp)
        if upload_resp.status_code >= 400:
            body = upload_resp.text.strip()[:500]
            raise RuntimeError(f"upload {fname} failed with HTTP {upload_resp.status_code}: {body}")
        return upload_resp.json()

    resp = _retry(
        f"upload {fname}",
        upload,
        retries=RETRY_COUNT,
        delay=RETRY_DELAY,
    )

    elapsed = time.time() - t0
    speed = fsize / elapsed / 1024 / 1024 if elapsed > 0 else 0
    print(f"\n  done in {elapsed:.1f}s ({speed:.1f} MB/s)")
    print(f"  id: {resp['id']}")


def cmd_get(args):
    svc = _service()
    folder_id = _find_folder(svc, args.folder)

    if args.name:
        q = f"name='{_q(args.name)}' and '{folder_id}' in parents and trashed=false"
    else:
        q = f"'{folder_id}' in parents and trashed=false and name contains '.bundle'"

    r = _request(
        f"list file {args.name or '(latest)'}",
        svc,
        'GET',
        f'{API_BASE}/files',
        params={
            'q': q,
            'spaces': 'drive',
            'fields': 'files(id,name,size,createdTime)',
            'orderBy': 'createdTime desc',
            'pageSize': 1,
        },
    ).json()
    items = r.get('files', [])

    if not items:
        print("nothing found")
        sys.exit(1)

    target = items[0]
    fname = target['name']
    fid = target['id']
    fsize = int(target.get('size', 0))

    dest_dir = args.dest or '.'
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, fname)

    print(f"getting {fname} ({fsize / 1024 / 1024:.1f} MB)...")
    t0 = time.time()

    resp = _request(
        f"download {fname}",
        svc,
        'GET',
        f'{API_BASE}/files/{fid}',
        params={'alt': 'media'},
        stream=True,
        timeout=(30, 600),
    )
    downloaded = 0
    with open(dest_path, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if fsize:
                pct = int(downloaded / fsize * 100)
                bar = '#' * (pct // 3) + '-' * (33 - pct // 3)
                print(f"\r  [{bar}] {pct}%", end='', flush=True)

    elapsed = time.time() - t0
    speed = fsize / elapsed / 1024 / 1024 if elapsed > 0 else 0
    print(f"\n  done in {elapsed:.1f}s ({speed:.1f} MB/s)")
    print(f"  saved: {dest_path}")
    print(dest_path)


def cmd_ls(args):
    svc = _service()
    folder_id = _find_folder(svc, args.folder)
    q = f"'{folder_id}' in parents and trashed=false"
    r = _request(
        f"list folder {args.folder}",
        svc,
        'GET',
        f'{API_BASE}/files',
        params={
            'q': q,
            'spaces': 'drive',
            'fields': 'files(id,name,size,createdTime)',
            'orderBy': 'createdTime desc',
            'pageSize': 20,
        },
    ).json()
    items = r.get('files', [])
    if not items:
        print("(empty)")
        return
    for f in items:
        sz = int(f.get('size', 0))
        ts = f.get('createdTime', '')[:19].replace('T', ' ')
        print(f"  {ts}  {sz/1024/1024:6.1f}M  {f['name']}")


def cmd_setup(args):
    print("=== Store Setup ===")
    print()
    print("1. Install Python deps:")
    print("   pip install google-auth google-auth-oauthlib requests")
    print()
    print(f"2. Place credentials.json in:")
    print(f"   {CRED_PATH}")
    print()
    print("   Get it from: Google Cloud Console > APIs > Credentials > OAuth 2.0 Client")
    print()
    print("3. Generate token (run on a machine with browser):")
    print(f"   python3 {os.path.join(SCRIPT_DIR, 'gen_token.py')}")
    print()
    print(f"4. Place token.json in:")
    print(f"   {TOKEN_PATH}")
    print()

    if os.path.exists(TOKEN_PATH):
        print(f"token.json: FOUND")
    else:
        print(f"token.json: NOT FOUND")
    if os.path.exists(CRED_PATH):
        print(f"credentials.json: FOUND")
    else:
        print(f"credentials.json: NOT FOUND")

    try:
        from google.oauth2.credentials import Credentials  # noqa: F401
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        import requests  # noqa: F401
        print("python deps: OK")
    except ImportError:
        print("python deps: NOT INSTALLED")


def main():
    parser = argparse.ArgumentParser(description='File store utility')
    sub = parser.add_subparsers(dest='cmd')

    p_put = sub.add_parser('put')
    p_put.add_argument('file')
    p_put.add_argument('--folder', default=DEFAULT_FOLDER)

    p_get = sub.add_parser('get')
    p_get.add_argument('name', nargs='?', default=None)
    p_get.add_argument('--folder', default=DEFAULT_FOLDER)
    p_get.add_argument('--dest', default=None)

    p_latest = sub.add_parser('latest')
    p_latest.add_argument('--folder', default=DEFAULT_FOLDER)
    p_latest.add_argument('--dest', default=None)

    p_ls = sub.add_parser('ls')
    p_ls.add_argument('--folder', default=DEFAULT_FOLDER)

    sub.add_parser('setup')

    args = parser.parse_args()

    if args.cmd == 'put':
        cmd_put(args)
    elif args.cmd == 'get':
        cmd_get(args)
    elif args.cmd == 'latest':
        args.name = None
        cmd_get(args)
    elif args.cmd == 'ls':
        cmd_ls(args)
    elif args.cmd == 'setup':
        cmd_setup(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
