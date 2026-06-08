# Security Policy

## Supported Versions

This project is pre-1.0 and experimental. Security fixes target the `main`
branch.

## Reporting A Vulnerability

Please open a private GitHub security advisory when possible. If that is not
available, contact the repository owner before publishing details.

Do not include real Kaggle tokens, OAuth credentials, private datasets, or
competition-only data in public issues.

## Credential Handling

- Kaggle authentication should use `kar auth` or Kaggle's supported local
  credential mechanisms.
- Do not commit `.env`, `kaggle.json`, OAuth tokens, browser cookies, or local
  credential caches.
- Do not print credentials in logs, reports, issue comments, or pull requests.

## Data And Submission Safety

- Kaggle datasets, generated features, models, predictions, submissions, and
  notebook caches are ignored by git by default.
- Real Kaggle submissions consume quota and may affect competition history; they
  require explicit user intent.
- Generated reports should avoid leaking private competition data unless that
  data is allowed to be shared under the competition rules.
