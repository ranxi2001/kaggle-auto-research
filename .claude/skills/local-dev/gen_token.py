#!/usr/bin/env python3
"""
Run on a machine with a browser to generate token.json.
pip install google-auth-oauthlib
"""
import os
import sys

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    print("pip install google-auth-oauthlib")
    sys.exit(1)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_PATH = os.path.join(SCRIPT_DIR, 'credentials.json')
TOKEN_PATH = os.path.join(SCRIPT_DIR, 'token.json')

if not os.path.exists(CRED_PATH):
    print(f"error: {CRED_PATH} not found")
    sys.exit(1)

flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH, SCOPES)
creds = flow.run_local_server(port=0)

with open(TOKEN_PATH, 'w') as f:
    f.write(creds.to_json())

print(f"token saved to: {TOKEN_PATH}")
