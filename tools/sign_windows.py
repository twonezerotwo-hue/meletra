"""Authenticode-sign (and verify) the Windows release files - only with a legitimate code-signing credential.

    python tools/sign_windows.py FILE [FILE ...] [--verify-only] [--json out.json]

Credentials (from the environment, never from a file in the repository):
    MLT_SIGN_PFX + MLT_SIGN_PFX_PASSWORD    a .pfx / .p12 certificate file and its password
    MLT_SIGN_THUMBPRINT                     SHA-1 thumbprint of a certificate in the Windows certificate store (also hardware tokens / cloud HSM KSPs)
    MLT_SIGN_TIMESTAMP                      RFC 3161 time-stamp server (default http://timestamp.digicert.com)

Without a credential nothing is signed and nothing is pretended: the tool prints
    WINDOWS SIGNING: BLOCKED - CODE SIGNING CREDENTIAL REQUIRED
and the one action that unblocks it, and exits with 3. A self-signed certificate is never created: it would not make the installer trusted and would only mislead. A valid signature
also does not promise that Microsoft SmartScreen stays silent: SmartScreen reputation builds with downloads (an EV certificate gives it immediately); the tool says nothing about it.
``--verify-only`` reports the Authenticode status of the files as they are (no credential needed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BLOCKED = "WINDOWS SIGNING: BLOCKED - CODE SIGNING CREDENTIAL REQUIRED"
ACTION = ("One action for the owner: buy or obtain a code-signing certificate from a public certificate authority (an EV certificate or a cloud / token based one gives SmartScreen "
          "trust at once), then export MLT_SIGN_PFX and MLT_SIGN_PFX_PASSWORD, or MLT_SIGN_THUMBPRINT for a certificate in the Windows store, and run this tool again.")
DEFAULT_TIMESTAMP = "http://timestamp.digicert.com"


def find_signtool() -> str | None:
    found = shutil.which("signtool")
    if found:
        return found
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin"
    candidates = sorted(kits.glob("*/x64/signtool.exe"), reverse=True) if kits.is_dir() else []
    return str(candidates[0]) if candidates else None


def credential(env=os.environ) -> dict | None:
    pfx, thumb = env.get("MLT_SIGN_PFX"), env.get("MLT_SIGN_THUMBPRINT")
    if pfx and Path(pfx).is_file():
        return {"kind": "pfx", "pfx": pfx, "password": env.get("MLT_SIGN_PFX_PASSWORD", "")}
    if thumb:
        return {"kind": "thumbprint", "thumbprint": thumb.replace(" ", "")}
    return None


def sign_command(signtool: str, cred: dict, file: Path, timestamp: str = DEFAULT_TIMESTAMP) -> list[str]:
    cmd = [signtool, "sign", "/fd", "sha256", "/td", "sha256", "/tr", timestamp]
    if cred["kind"] == "pfx":
        cmd += ["/f", cred["pfx"]] + (["/p", cred["password"]] if cred["password"] else [])
    else:
        cmd += ["/sha1", cred["thumbprint"]]
    return cmd + [str(file)]


def authenticode_status(file: Path) -> dict:
    """Status and signer of a file, read with PowerShell's Get-AuthenticodeSignature (works without signtool)."""
    quoted = str(file).replace("'", "''")
    script = (f"$s = Get-AuthenticodeSignature -LiteralPath '{quoted}'; "
              "[pscustomobject]@{status = [string]$s.Status; signer = if ($s.SignerCertificate) { $s.SignerCertificate.Subject } else { '' }; "
              "timestamp = if ($s.TimeStamperCertificate) { 'yes' } else { 'no' } } | ConvertTo-Json -Compress")
    if sys.platform != "win32":                           # Authenticode is read by Windows PowerShell; elsewhere nothing is verified, and nothing is claimed
        return {"status": "Unknown", "error": "Authenticode signatures can only be read on Windows"}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=600)
        info = json.loads(out.stdout.strip() or "{}") or {}
        return info if info.get("status") else {**info, "status": "Unknown"}
    except Exception as exc:                              # noqa: BLE001
        return {"status": "Unknown", "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    files = [f for f in a.files if f.is_file()]
    missing = [str(f) for f in a.files if not f.is_file()]
    report: dict = {"files": {}, "missing": missing}
    if not a.verify_only:
        cred = credential()
        if cred is None:
            print(BLOCKED)
            print(ACTION)
            report["signing"] = "BLOCKED"
            for f in files:
                report["files"][str(f)] = authenticode_status(f)
            if a.json:
                a.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
            return 3
        tool = find_signtool()
        if tool is None:
            print("WINDOWS SIGNING: BLOCKED - signtool.exe REQUIRED (install the Windows 10/11 SDK 'Signing Tools for Desktop Apps')")
            report["signing"] = "BLOCKED"
            return 3
        timestamp = os.environ.get("MLT_SIGN_TIMESTAMP", DEFAULT_TIMESTAMP)
        for f in files:
            proc = subprocess.run(sign_command(tool, cred, f, timestamp), capture_output=True, text=True)
            report["files"][str(f)] = {"signtool_exit": proc.returncode}
            if proc.returncode != 0:
                print(f"signing FAILED for {f}: {(proc.stderr or proc.stdout).strip()[:200]}")
        report["signing"] = "DONE"
    for f in files:
        status = authenticode_status(f)
        report["files"].setdefault(str(f), {}).update(status)
        print(f"{f.name}: {status.get('status')}  {status.get('signer', '')}")
    bad = [f for f, s in report["files"].items() if s.get("status") != "Valid"]
    report["all_valid"] = not bad and not missing
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.verify_only:
        return 0 if report["all_valid"] else 1
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
