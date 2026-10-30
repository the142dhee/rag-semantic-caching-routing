# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Reporting a Vulnerability

Security is paramount in enterprise and AI engineering systems. If you discover a security vulnerability or credential exposure risk:

1. **Do NOT open a public GitHub issue.**
2. Send an email to the repository maintainer with detailed reproduction steps, proof of concept, and affected components.
3. We acknowledge reports within 48 hours and work with you to patch the vulnerability responsibly before public disclosure.

---

## Security Practices Implemented

- **Secret Sanitization**: Automated redactors scrub API keys, bearer tokens, and credentials from logs and telemetry traces (`app/services/tracing.py`).
- **Authentication**: Optional API Key header verification (`X-API-Key`) protecting mutating operations.
- **Rate Limiting**: Sliding-window IP throttling to mitigate Denial of Service and API quota exhaustion.
- **Least Privilege Containers**: Docker containers run under a non-root `appuser`.
- **Safe Parsing**: In-memory stream validation and error trapping for malformed or corrupted PDF uploads.
