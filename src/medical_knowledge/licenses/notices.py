"""Third-party notices that travel with a knowledge database (``<db>.NOTICE.txt``): who the data come from, under which licence, and the attribution text each
licence asks for. Written by the builder next to the manifest; a public installer ships this file with the database."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
CURRENCY = ("MeSH data are a snapshot of the release named above and may not reflect the most current MeSH. Courtesy of the U.S. National Library of Medicine.")


def notice_text(manifest: dict) -> str:
    public = manifest.get("profile") == "public"
    lines = [f"{manifest.get('name', 'MedicalKnowledge')} {manifest.get('build_version', '')} - {'THIRD-PARTY NOTICES' if public else 'PRIVATE DATABASE - NOT FOR REDISTRIBUTION'}",
             f"profile: {manifest.get('profile')}   built: {manifest.get('build_date')}   file: {manifest.get('file')}   sha256: {manifest.get('sha256')}", ""]
    if not public:
        private = [s["source_id"] for s in manifest.get("sources", []) if not s.get("redistributable")]
        lines += [f"This database contains sources that must not be redistributed: {', '.join(private) or 'none listed'}.", ""]
    for s in manifest.get("sources", []):
        lines += [f"== {s['name']} ({s['source_id']}) version {s.get('version') or 'n/a'}",
                  f"   licence: {s.get('license_name')}   status: {s.get('license_status')}   redistribution: {'allowed' if s.get('redistributable') else 'NOT allowed'}"]
        if s.get("license_url"):
            lines.append(f"   licence text: {s['license_url']}")
        if s.get("attribution"):
            lines.append(f"   attribution: {s['attribution']}")
        if s["source_id"] == "MESH":
            lines.append(f"   currency notice: {CURRENCY}")
        if s["source_id"] == "SCOWL":
            copyright_file = HERE / "SCOWL_COPYRIGHT.txt"
            if copyright_file.is_file():
                lines += ["   SCOWL Copyright (must accompany every redistribution):", ""] + ["      " + t for t in copyright_file.read_text(encoding="utf-8", errors="replace").splitlines()]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_notice(db_path: Path, manifest: dict) -> Path:
    target = Path(str(db_path) + ".NOTICE.txt")
    target.write_text(notice_text(manifest), encoding="utf-8")
    return target
