"""Build a Medical Knowledge database.

    python tools/build_medical_knowledge.py --profile public --output data/medical_knowledge_public.db
    python tools/build_medical_knowledge.py --profile full   --output data/medical_knowledge_full.db [--umls PATH_TO_META]

``public``  only sources that are ALLOW in ``licenses/registry.json`` AND listed in ``licenses/public_allowlist.json``; offering anything else FAILS the build.
``full``    everything the owner can use locally (licensed UMLS data is read from a local copy, never downloaded).
``fixture`` the small test database (curated vocabulary + test fixtures).

Prints a summary and writes ``<output>.manifest.json`` (version, sources and their licenses, counts, SHA-256).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical_knowledge.ingest.builder import KnowledgeBuilder  # noqa: E402
from src.medical_knowledge.licenses.policy import LicenseViolation  # noqa: E402


def sources_for(args) -> list:
    from src.medical_knowledge.sources.curated import CuratedSource
    from src.medical_knowledge.sources.fixture import FixtureSource

    chosen = [CuratedSource(use_confusables=not args.no_confusables), FixtureSource()]
    if args.umls:
        from src.medical_knowledge.sources.umls import UmlsSource

        chosen.append(UmlsSource(Path(args.umls)))
    if args.pubmed:
        from src.medical_knowledge.sources.pubmed import PubmedSource

        chosen.append(PubmedSource(Path(args.pubmed)))
    for extra in args.source or []:
        from src.medical_knowledge.sources import registry

        chosen.append(registry.create(extra, Path(args.data_dir) if args.data_dir else None))
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["public", "full", "fixture"], default="fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-version", default=None, help="e.g. 2026.10")
    parser.add_argument("--umls", help="folder of a locally licensed UMLS release (META/ with MRCONSO.RRF ...)")
    parser.add_argument("--source", action="append", help="additional open source by id (mesh, go, hpo, chebi, hgnc)")
    parser.add_argument("--data-dir", help="folder with downloaded open sources")
    parser.add_argument("--pubmed", help="mined PubMed terms file (tools/mine_pubmed.py): UNVERIFIED candidates, private builds only")
    parser.add_argument("--no-confusables", action="store_true", help="leave out the known ASR mishearings (generalisation tests)")
    args = parser.parse_args(argv)
    started = time.time()
    builder = KnowledgeBuilder(args.output, args.profile, args.build_version)
    try:
        for source in sources_for(args):
            stats = source.export(builder)
            print(f"  {stats.source_id}: {stats.concepts} concepts, {stats.terms} terms, {stats.relations} relations, {stats.abbreviations} abbreviations, "
                  f"{stats.cues} cues, {stats.confusions} confusions")
        manifest = builder.finalize()
    except LicenseViolation as exc:
        print(f"BUILD FAILED (license): {exc}")
        return 3
    except Exception:
        builder.abort()
        raise
    c = manifest["counts"]
    print(f"{manifest['name']} {manifest['build_version']} [{manifest['profile']}]: {c['concepts']} concepts, {c['terms']} terms, {c['abbreviations']} abbreviations, "
          f"{c['relationships']} relationships; {manifest['size_bytes'] / 1e6:.1f} MB; {time.time() - started:.1f} s; sha256 {manifest['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
