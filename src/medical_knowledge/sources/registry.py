"""Names of the open sources the build tool can add (``--source mesh``) and where their files are expected."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import TerminologySource

FILES = {"mesh": ("desc2026.gz", "desc2026.xml"), "go": ("go-basic.obo",), "hpo": ("hp.obo",), "chebi": ("chebi_lite.obo",), "hgnc": ("hgnc_complete_set.txt",), "scowl": ("scowl",)}


def locate(name: str, data_dir: Optional[Path]) -> Path:
    folder = Path(data_dir) if data_dir else Path(__file__).resolve().parents[3] / "data" / "open_raw"
    for candidate in FILES[name]:
        if (folder / candidate).is_file() or (name == "scowl" and (folder / candidate / "final").is_dir()):
            return folder / candidate
    if name == "mesh":
        found = sorted(folder.glob("desc20*.gz")) + sorted(folder.glob("desc20*.xml"))
        if found:
            return found[-1]
    return folder / FILES[name][0]


def create(name: str, data_dir: Optional[Path] = None, limit: int = 0) -> TerminologySource:
    name = name.lower()
    if name not in FILES:
        raise ValueError(f"unknown source '{name}' (known: {', '.join(FILES)})")
    path = locate(name, data_dir)
    if name == "mesh":
        from .mesh import MeshSource
        return MeshSource(path, limit)
    if name == "scowl":
        from .scowl import ScowlSource
        return ScowlSource(path)
    if name == "hgnc":
        from .hgnc import HgncSource
        return HgncSource(path, limit=limit)
    from .obo import ChebiSource, GoSource, HpoSource
    return {"go": GoSource, "hpo": HpoSource, "chebi": ChebiSource}[name](path, limit)
