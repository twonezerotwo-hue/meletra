# Transcript inbox (reference backend)

The small server behind **Settings -> Privacy & Data** of MELETRA. It is NOT part of the desktop program and is not deployed by this repository: the owner runs it
somewhere with HTTPS (any host that can run Python 3.11 or a container) and puts its address into `config/sharing.json` -> `"endpoint"` before building a release.
Until that address is set the program ships with sharing switched on by default but sends nothing (the Settings page says so).

```bash
OWNER_NOTIFICATION_EMAIL=owner@example.org SMTP_HOST=smtp.example.org SMTP_USER=... SMTP_PASSWORD=... MAIL_FROM=... python -m backend.transcript_inbox.server
```

* `POST /v1/transcripts` - the JSON of `src/sharing/payload.py` (unknown fields are refused, 3 MB at most, 30 requests per hour per address and per installation).
* The recipient and the mail credentials are environment variables of the SERVER. The desktop binary holds none of them.
* Received transcripts are stored in `INBOX_DIR` (one JSON per transcript, mode 0600) before any mail is attempted: a mail problem never loses a transcript.
