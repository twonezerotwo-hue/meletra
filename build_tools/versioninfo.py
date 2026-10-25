"""Windows version resource for the .exe (shown in Explorer -> Properties -> Details)."""

from __future__ import annotations

from pathlib import Path


def version_tuple(version: str) -> tuple[int, int, int, int]:
    """Numbers of a version for the exe resource; a pre-release suffix (``0.2.0-rc1``) is ignored, short forms ("2") are padded."""
    from .layout import split_version

    try:
        return split_version(version)[0]
    except ValueError:
        parts = [int(p) for p in version.split("-")[0].split(".")[:4]]
        parts += [0] * (4 - len(parts))
        return tuple(parts)  # type: ignore[return-value]


def version_file_text(version: str, company: str, product: str, description: str, exe_name: str, copyright_text: str = "") -> str:
    v = version_tuple(version)
    dotted = ".".join(str(p) for p in v)
    shown = version if version.count(".") >= 2 else dotted
    copyright_text = copyright_text or f"Copyright (C) {company}"
    fields = [("CompanyName", company), ("FileDescription", description), ("FileVersion", dotted), ("InternalName", product),
              ("LegalCopyright", copyright_text), ("OriginalFilename", exe_name), ("ProductName", product), ("ProductVersion", shown)]
    strings = ",\n        ".join(f"StringStruct({k!r}, {v_!r})" for k, v_ in fields)
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers={v}, prodvers={v}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
        {strings}
    ])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def write_version_file(destination: Path, version: str, company: str, product: str, description: str, exe_name: str) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(version_file_text(version, company, product, description, exe_name), encoding="utf-8")
    return destination
