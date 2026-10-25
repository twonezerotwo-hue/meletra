"""The lecture archive as ONE thing: where a lecture's files are, what they hold, and what is known about the lecture.

Before 4.1.2 every screen (Recent lectures, the result screen, Read transcript, Study, Technical details, Translate) found a lecture's transcript by its own rule. A finished job's record
named the folder the student CHOSE (the "Medical Transcription" folder), the lecture's own folder (its "FIXTURE_0005" sub-folder) was only in the job summary, so a perfectly good
transcript was reported as missing. This package is the single place that knows:

* ``artifacts``  - which files of a lecture folder are transcripts, translations, study material (current and legacy names);
* ``transcript`` - the text of a lecture, read from whichever of those files exists, with the times it really has;
* ``resolver``   - record -> lecture folder, and the facts shown for a lecture (words, duration, language, status) with an honest "unknown" where nothing is recorded;
* ``reconcile``  - the safe, repeatable repair that makes lecture folders on disk and archive records agree.
"""
