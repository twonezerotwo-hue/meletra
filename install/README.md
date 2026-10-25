# install/ - installers for COLLABORATORS of the private repository

These two scripts read the **private** repository's own GitHub releases (they need `gh auth login` or a `GH_TOKEN` that can read the repository). They are for people who
work on the project, not for users.

**To install MELETRA, do not use this folder.** Use the public download page or the one-command installer from the README ("Install MELETRA"):
they come from the public distribution repository `twonezerotwo-hue/whispermedical-downloads`, need no GitHub account, and are driven by its `release-manifest.json`.

| file | what it does |
|---|---|
| `install-windows.ps1` | finds the newest Windows release of the private repository, downloads it, checks its SHA-256, runs the installer (or unzips the portable version) |
| `install-macos.sh` | the same for the Intel Mac DMG of the private repository |
