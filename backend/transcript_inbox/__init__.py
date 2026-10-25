"""Reference backend for the transcripts that MELETRA shares (Settings -> Privacy & Data). Not part of the desktop program.

The desktop client posts a small JSON over HTTPS (see src/sharing/payload.py). This server validates it against the same allow-list, stores it, and notifies the owner by
e-mail. The e-mail address and the mail credentials exist ONLY here, as environment variables of the server:

    OWNER_NOTIFICATION_EMAIL   who is notified (for example the owner's address)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, MAIL_FROM        the mail account the server sends from
    INBOX_DIR                  where received transcripts are stored (default ./inbox)
    ATTACH_TRANSCRIPT          1 (default) = the transcript is attached to the notification

The desktop program contains no mail address, no mail password and no mail key.
"""
