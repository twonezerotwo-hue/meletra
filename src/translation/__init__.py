"""Output Language: the finished transcript in the language the student asked for, next to the spoken-language transcript (which is never changed).

* same language as spoken       -> nothing to do;
* output English                -> Whisper's own translate task (already on the computer: no extra model);
* any other output language     -> an offline translation model (M2M100, MIT licence) that is installed on request (Settings -> Models); without it the
                                   spoken transcript is kept and the result says that the translation is not installed.

Files written next to the transcript (the ``local_*`` files stay exactly as they were):
    translated_<code>_transcript.txt / .srt / .json          e.g. translated_en_transcript.txt
"""

from __future__ import annotations

METHOD_NONE = "none"                # no translation needed (same language / no output language asked for)
METHOD_WHISPER = "whisper"          # Whisper task=translate (English output only)
METHOD_M2M100 = "m2m100"            # the offline translation model
METHOD_MISSING = "model_missing"    # a translation was asked for but its model is not installed: the spoken transcript is all there is
METHOD_FAILED = "failed"            # the translation stopped with an error: the spoken transcript is complete, the translation is not
METHODS = (METHOD_NONE, METHOD_WHISPER, METHOD_M2M100, METHOD_MISSING, METHOD_FAILED)
