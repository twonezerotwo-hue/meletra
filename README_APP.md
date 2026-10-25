# MELETRA — desktop app (written for 3.0, still valid for 4.1)

Turn **any file** into **text** — a lecture recording, a video, a PDF, a picture of a slide, a text file — and the text into **speech** or
**study material** (summaries, flashcards, exam questions, an interactive Study Mode) — **on your own computer**. Nothing is uploaded
unless you choose a cloud option yourself. No Python, FFmpeg or CUDA installation is needed: everything is inside the program.

> Kısa özet (TR): Programı açın → dosyaları sürükleyin ya da **SELECT FILES** (ses, video, PDF, resim, metin; ses/video en fazla 10 tane) →
> **CONVERT TO TEXT**. Bitince: **Read** (oku), **Listen to Text** (sesli oku), **Create Study Material** (özet, flashcard, soru, çalışma
> modu). Konuşma tanıma modeli (Whisper Large-v3) bilgisayarda zaten varsa (örn. Hugging Face önbelleği) otomatik bulunur: seçmeniz,
> kopyalamanız ya da yeniden indirmeniz gerekmez. Her şey bu bilgisayarda çalışır.

## What is new in 3.0

* **One start screen for every file.** The kind is recognised from the file's content (not just its name): audio, video, PDF, pictures
  (png, jpg, webp, tiff, bmp) and text (txt, md). Pick several files at once.
* **Up to 10 recordings in a row.** They are transcribed one after another with the same engine as before (pause, resume, cancel; a file
  that fails does not stop the others). Results go to `Medical Transcription\Batch_<time>\<name>\` and are named after the source
  (`Lecture_01_raw.txt`, `Lecture_01_clean.txt`, `Lecture_01.srt`, `Lecture_01.json` ...).
* **PDFs and pictures to text.** Text that is inside a PDF is used as it is ("Embedded PDF Text", no confidence number is invented). Only
  pages without text (scans) and pictures go through text recognition (OCR) - Windows' own recognition, nothing to install - and show their
  real confidence when the engine reports one.
* **Listen to Text.** Reads a transcript or a document aloud with the voices already installed on the computer (English and Turkish out
  of the box on typical Windows; German and Hungarian when such a voice is installed). The result can also be saved as WAV or MP3 by the
  program's text-to-speech module.
* **The speech model is found automatically** in the usual Hugging Face cache folders (and the program's own folders); manual import
  remains under Settings → Models.
* The transcription engine itself (Whisper Large-v3, GPU/CPU, pause/resume, repair of gaps and loops, medical context) is unchanged;
  the study features of 0.2 are unchanged and sit behind **Create Study Material**.

## Which package?

| File | What it is | Needs internet? |
|---|---|---|
| `MELETRA-Setup-Windows-x64.exe` | Windows installer: Program Files, Start Menu entry, optional desktop shortcut, uninstaller. Models, settings and logs are **kept** when you uninstall (you are asked). | Only for the first model download |
| `MELETRA-Portable-Windows-x64.zip` | Windows, unzip anywhere (USB drive too). Settings, models and logs stay in a `data` folder next to the program. | Only for the first model download |
| `MedicalLectureTranscriber-macOS-AppleSilicon.dmg` | macOS (M1 or newer): open the disk image and drag the app into *Applications*. Built on a Mac only (see BUILDING.md); **not part of a Windows-only build.** | Only for the first model download |
| `MedicalLectureTranscriber-macOS-Intel.dmg` | macOS on Intel, secondary (slower: the CPU does the speech recognition). | Only for the first model download |

**Windows 10 (1809+) or 11, 64-bit.** Minimum 8 GB RAM, 6 GB free disk (10 GB with the model). Recommended: 16 GB RAM, an NVIDIA GPU with
6 GB+ VRAM (about 4× faster than real time on an RTX 2070). Without a GPU the CPU is used automatically (slower).
**macOS 12+**, Apple Silicon recommended (the speech model runs on the Mac's GPU).

**Windows SmartScreen / macOS Gatekeeper:** the program is not code-signed yet. Windows: *More info → Run anyway*. macOS: right-click the
app, choose *Open*, then *Open* (needed the first time only). A signed + notarized Mac build needs an Apple Developer ID certificate,
which this project does not have yet.

## First start

A short check looks at your computer (Windows: RAM, CPU, GPU, CUDA, disk — Mac: Apple Silicon or Intel, RAM, disk, the speech engine,
whether the model is installed) and offers **DOWNLOAD RECOMMENDED MODEL** (Whisper Large-v3, about 3 GB, resumable — you can close the
program and continue later). Already have the model? *I already have the model — import…*. You can skip it and do it later.

## The simple flow

1. **Drop your lecture** into the window (or click it): mp3, wav, m4a, aac, flac, ogg, opus, mp4, mov, mkv, webm. Optionally
   **Add Lecture PDF** — the slides help with difficult terms and make the study material better.
2. Press **TRANSCRIBE**. The screen says *Transcribing your lecture…* with a progress bar, the percent and
   *Estimated time remaining*, a **Pause** button (it saves your progress; close the program and continue later, even after a
   crash) and, small, the detected *Language* and *Subject*.
3. **Your lecture is ready.** Choose **Read Transcript** (a reader with search, clean / timed / raw views, copy), **Study Lecture**,
   or **Open Files**. The result shows Transcript, Summary, Flashcards and Questions; technical reports are behind **Technical details**.

Language and subject are detected automatically (the program listens to windows spread over the whole recording, not just the first
minutes); set a fixed language or subject in Settings → General if you ever need to.

Output goes to `<folder of the recording>\Medical Transcription\<file name>\` unless you choose another folder in Settings → General. A
different recording with the same file name never overwrites an older result (`<file name> (2)`). You can reopen any finished lecture later
with **Open a finished lecture…** on the start screen (File menu too).

| File | Content |
|---|---|
| `local_clean_transcript.txt` | Readable text with conservative, dictionary-backed fixes and `[UNCERTAIN: …]` marks |
| `local_raw_transcript.txt` | Exactly what the speech model produced — **never edited** |
| `local_transcript_with_timestamps.txt`, `.srt`, `.vtt`, `.json` | Timed versions |
| `medical_terms_detected.txt`, `possible_medical_errors.txt` | Medical terms heard; suspected mis-hearings with evidence (nothing is changed automatically) |
| `quality_report.json`, `metadata.json`, `processing.log` | Coverage, repeated-text checks, timing, settings used |
| `study\` | The study material of this lecture (see below) |

The transcript is a draft. Check drug names, doses and numbers against the source before relying on them.

## Studying a lecture

**Study Lecture** opens a workspace with six tabs. The first time it makes the material (usually a few seconds with the built-in method);
afterwards it opens at once.

| Tab | What you get |
|---|---|
| **Overview** | What it was made from (slides + transcript / transcript / slides only), counts, your progress, buttons to practise |
| **Summary** | Quick, Detailed, High-Yield, Mechanisms / Pathways, Clinical Connections, Lecturer Emphasis, Exam-Relevant — in lecture order, with the lecture's own terminology |
| **Key Concepts** | The main terms and the important definitions |
| **Flashcards** | 12 types (definition, concept, mechanism, cause→effect, structure→function, clinical, comparison, pathway, enzyme, drug, gene/protein, exam-style recall), easy / medium / hard. Show answer, then **Know / Unsure / Don't know**; progress is remembered |
| **Questions** | Multiple choice (5 options), true/false, short answer, explain why, clinical vignette, mechanism. The correct answer appears only after you answer |
| **Study Mode** | One question at a time: answer → correct / incorrect → explanation → next. At the end: score, weak topics, strong topics, what to review. Weak questions come first next time |

**Every item shows where it comes from** (*Slide 14*, *Transcript 12:30-12:58*); click it to read the passage. Study material is
**source-bound**: it only says what your lecture says. Medical facts that are not in your lecture are not added, and if you ask the program
to explain something the lecture does not cover, it says so instead of guessing. Add the lecture PDF later with **Add lecture PDF…**.
No recording at all? **Study from a PDF only…** on the start screen.

**AI Study (Settings):** the default *Built-in* method works offline and quotes the lecture. For more fluent summaries and questions you can
choose a **local AI model** (Ollama, or any OpenAI-compatible server on this computer — nothing leaves the PC) or a **cloud model**
(then the lecture *text* — never the audio — is sent to that provider; the screen says so). API keys are stored in the Windows Credential
Manager / macOS Keychain. If the chosen AI is not available the built-in method is used and the overview says so; the transcript is never
affected. More in `docs/STUDY_SYSTEM.md`.

## Settings

* **General** — output folder, theme, default language, subject and mode, *Show advanced options*.
* **AI Study** — study method, model, server address, key, language, number of cards / questions, *Test this study method*.
* **Models** — installed models, download / resume / verify (SHA-256) / delete / import a model folder / add a search folder (e.g. a USB drive), disk usage.
* **Cloud** — LOCAL (default: no audio leaves the computer) or DEEPGRAM (optional: audio is uploaded). The key is stored in the system's secure store, never in a file or a log.
* **Advanced** (only after *Show advanced options*) — *Window style* (Simple / Classic technical window of 0.1.1), **Performance** (device,
  compute type, CPU threads, beam size, voice activity detection, chunk size, GPU-memory warnings), **Medical** (context, hotwords, slides, output formats),
  **Diagnostics** (system, GPU/engine, libraries, FFmpeg, models, paths; **COPY DIAGNOSTICS** for support). *Help → Advanced Diagnostics…* opens it directly.

## Where files are kept

| What | Windows (installed) | Windows (portable) | macOS |
|---|---|---|---|
| Models, settings, unfinished work | `%LOCALAPPDATA%\MedicalLectureTranscriber\` | `data\` next to the program | `~/Library/Application Support/MedicalLectureTranscriber/` |
| Caches, logs | inside the folder above | inside `data\` | `~/Library/Caches/…` and `~/Library/Logs/…` (`MedicalLectureTranscriber`) |
| Transcripts and study files | the output folder you chose | the same | the same |
| API keys (optional) | Windows Credential Manager | the same | macOS Keychain |

`logs\app.log` (and `app.previous.log`) and `logs\job-<id>.log` are the first place to look when something goes wrong.

## GPU, CUDA, Apple Silicon and CPU

Windows without an NVIDIA GPU (or when CUDA cannot be used): the start page says *CUDA acceleration unavailable* with the reason and the CPU
is used. On an Apple Silicon Mac the optimized speech engine uses the Mac's GPU; on an Intel Mac the CPU is used (slower). Transcripts from the
two engines can differ by a few words (the Mac engine decodes greedily; the Windows engine uses beam search).

## Troubleshooting

| Problem | What to do |
|---|---|
| "CUDA acceleration unavailable" although you have an NVIDIA GPU | Update the NVIDIA driver; Settings → Advanced → Diagnostics → COPY DIAGNOSTICS shows the reason. The CPU is used meanwhile. |
| "GPU memory is low" | Close programs that use the GPU (browser tabs with video, Ollama, games) or choose Fast mode. |
| The study material says the AI was not available | The built-in method was used (nothing is lost). Settings → AI Study → *Test this study method* tells what is missing (for Ollama: `ollama pull <model>`). |
| *This PDF has no text* | The slides are pictures (a scan). Use a text-based PDF. |
| Loops (the same sentence many times) in a transcript | They are found and re-decoded automatically in short pieces; what could not be repaired stays in the raw transcript and is flagged in `quality_report.json`. |
| Audio decoding problems | The program has its own decoder (PyAV) and a bundled FFmpeg; `MLT_DISABLE_FFMPEG=1` forces the built-in decoder only. |
| Model download interrupted | Press the button again: it resumes. Settings → Models → *Verify* checks every file. |
| The window closed during a transcription | Open the program, drop the same file, press **Continue where it stopped**. |
| Something looks different from the old window | Settings → Advanced → *Window style: Classic* brings back the technical window of version 0.1.1. |

## Uninstall

Windows: Settings → Apps → *MELETRA* → Uninstall (you are asked whether to also delete models, settings and logs; default: keep).
Portable packages: delete their folder. macOS: drag the app to the Bin; delete the folders above if you also want the models gone.

## Known limitations

* One transcription at a time.
* Whisper decodes one language per 30-second window; mixed Turkish/English lectures are best transcribed with *Language: Auto* and checked afterwards (the macOS engine has no per-window language switching).
* Speaker names and numbers (doses!) can be wrong; the Possible Medical Errors list is a hint, not a correction.
* The built-in study method is plain: it quotes your lecture. A local or cloud AI writes more fluent items but can be wrong — the program keeps only what the lecture supports, but read what you study.
* The optional Turkish medical model (Transformers) is not part of the packaged app; it works from a source install only.
* The macOS app was prepared on Windows and has **not been run on a Mac yet**; it is unsigned and not notarized.
