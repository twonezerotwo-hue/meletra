"""License policy of the knowledge base: what may enter which database.

Two databases are built from the same importers:

* ``full``    for private local development; may contain sources the owner is licensed to use locally (UMLS, SNOMED CT ...).
* ``public``  what the public installer may carry. Generated from an EXPLICIT allowlist (``public_allowlist.json``); a source is only in it after its
              license text was read and recorded in ``registry.json`` with status ALLOW. Nothing is included "because it is probably fine".

Status of a source (never guessed - a source without a registry entry is REVIEW_REQUIRED):

    ALLOW            the license text permits redistribution inside the product (attribution, if required, is shipped with it)
    PRIVATE_ONLY     usable for local development, must never be redistributed
    REVIEW_REQUIRED  not decided; behaves like PRIVATE_ONLY until somebody reads the license and records the decision
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
STATUSES = ("ALLOW", "PRIVATE_ONLY", "REVIEW_REQUIRED")
PROFILES = ("public", "full", "fixture")


class LicenseViolation(Exception):
    """A source that is not allowed in a public build was offered to it. The build fails; nothing is silently dropped."""


@dataclass(frozen=True)
class SourceLicense:
    source_id: str
    name: str
    license_name: str
    license_url: str = ""
    status: str = "REVIEW_REQUIRED"
    requires_credentials: bool = False
    attribution: str = ""
    verified_on: str = ""                     # date the license text was read
    evidence: str = ""                        # what the license text says, in one sentence (or "not read yet")
    notes: str = ""

    @property
    def redistribution_allowed(self) -> bool:
        return self.status == "ALLOW"


@lru_cache(maxsize=1)
def registry(path: str = str(HERE / "registry.json")) -> dict[str, SourceLicense]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, SourceLicense] = {}
    for row in data["sources"]:
        lic = SourceLicense(**row)
        if lic.status not in STATUSES:
            raise ValueError(f"{lic.source_id}: unknown status {lic.status}")
        out[lic.source_id] = lic
    return out


@lru_cache(maxsize=1)
def public_allowlist(path: str = str(HERE / "public_allowlist.json")) -> frozenset[str]:
    return frozenset(json.loads(Path(path).read_text(encoding="utf-8"))["sources"])


def license_of(source_id: str) -> SourceLicense:
    """The recorded license of a source; an unknown source is REVIEW_REQUIRED (= private).

    A sub-source ``PARENT/CHILD`` (one vocabulary of UMLS: ``UMLS/MSH``) has its own entry when it was reviewed; without one it takes the license of its parent.
    """
    reg = registry()
    if source_id in reg:
        return reg[source_id]
    if "/" in source_id and source_id.split("/", 1)[0] in reg:
        parent = reg[source_id.split("/", 1)[0]]
        return SourceLicense(source_id, f"{parent.name} / {source_id.split('/', 1)[1]}", parent.license_name, parent.license_url, parent.status,
                             parent.requires_credentials, parent.attribution, parent.verified_on, f"inherits from {parent.source_id}: {parent.evidence}", parent.notes)
    return SourceLicense(source_id, source_id, "unknown", status="REVIEW_REQUIRED", evidence="no registry entry")


def allowed_in_public(source_id: str) -> bool:
    lic = license_of(source_id)
    return lic.status == "ALLOW" and source_id in public_allowlist()


def check_public(source_ids: Iterable[str]) -> list[str]:
    """Problems that make a public build impossible (empty list = fine)."""
    problems = []
    for sid in sorted(set(source_ids)):
        lic = license_of(sid)
        if lic.status != "ALLOW":
            problems.append(f"{sid}: license status {lic.status} ({lic.license_name}) - not redistributable")
        elif sid not in public_allowlist():
            problems.append(f"{sid}: license is ALLOW but the source is not in public_allowlist.json")
    return problems


def require_for_profile(source_id: str, profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    if profile == "public":
        problems = check_public([source_id])
        if problems:
            raise LicenseViolation("; ".join(problems))
