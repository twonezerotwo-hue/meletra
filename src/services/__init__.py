"""Desktop-app services: everything the GUI needs that is not speech recognition.

The GUI never contains ASR logic. It talks to these services, and the services talk to the existing pipeline
(:mod:`src.pipeline`) through :mod:`src.services.transcription_service`.
"""
