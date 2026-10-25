"""Sharing of completed transcripts with MELETRA (Settings -> Privacy & Data). OFF by default: it is switched on only by the student, who is first shown what would be shared.

How it behaves (all of it is tested):
* the transcript is saved on this computer FIRST; sharing starts only after the job reports that it is complete (``SharingService.transcript_saved``);
* what is prepared is a small JSON (``payload.py``): transcript text, the medical corrected text, the lecture name, the languages, the program version, a random
  installation number and the time. Never audio, video, PDF, pictures, paths, user or computer names, hardware numbers;
* the JSON goes into a local queue (``queue.py``) and a background thread sends it over HTTPS to the developer's backend (``uploader.py``). The backend holds the
  mail settings; the program contains no mail address, mail password or mail key;
* a failure is silent: the item stays in the queue and is retried with growing pauses; it is visible only in Settings -> Privacy & Data -> Upload status;
* switching sharing off stops all future uploads at once and deletes everything still waiting (``SharingService.set_enabled(False)``);
* nothing is sent unless the student switched sharing on AND agreed to the notice (``share_notice_shown``), and never while no backend address is configured (``config/sharing.json``).
"""

from __future__ import annotations

SCHEMA = 1
MAX_PAYLOAD_BYTES = 3 * 1024 * 1024
