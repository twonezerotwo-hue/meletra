"""Import a LOCALLY LICENSED UMLS Metathesaurus into the private (full) Medical Knowledge database.

    python tools/import_umls.py --source <path-to-META-or-release-folder>             build data/medical_knowledge_full.db (UMLS + curated vocabulary)
    python tools/import_umls.py --source <path> --check                                only check that the files are usable
    python tools/import_umls.py --source <path> --stats                                count atoms / concepts / vocabularies / languages (no database)
    python tools/import_umls.py --help-license                                         where to get UMLS and what the license means for this project

Options: --languages ENG,LAT,GER,HUN,TUR   --sabs MSH,MSHGER,...   --no-relations   --max-related 30   --limit-concepts N   --output PATH

The importer does NOT download UMLS and does NOT bypass the license agreement. UMLS-derived data is written to the ``full`` database only; the public
build refuses it. Raw UMLS files are never committed to Git (``/data/`` and ``*.RRF`` are ignored).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical_knowledge.sources.umls import DEFAULT_LANGUAGES, HOW_TO_GET, UmlsSource, rows  # noqa: E402


def stats(src: UmlsSource) -> None:
    langs: Counter = Counter()
    sabs: Counter = Counter()
    cuis = set()
    n = 0
    for r in rows(src.files["MRCONSO.RRF"], 17):
        n += 1
        langs[r[1]] += 1
        sabs[r[11]] += 1
        cuis.add(r[0])
    print(f"MRCONSO.RRF: {n:,} atoms, {len(cuis):,} concepts, {len(sabs)} vocabularies, {len(langs)} languages")
    print("  languages:", ", ".join(f"{k} {v:,}" for k, v in langs.most_common(10)))
    print("  vocabularies:", ", ".join(f"{k} {v:,}" for k, v in sabs.most_common(12)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "medical_knowledge_full.db")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--sabs", default="")
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--max-related", type=int, default=30)
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--help-license", action="store_true")
    args = parser.parse_args(argv)
    if args.help_license or not args.source:
        print(HOW_TO_GET)
        return 0 if args.help_license else 2
    src = UmlsSource(args.source, args.languages.split(","), [s for s in args.sabs.split(",") if s] or None, not args.no_relations, args.max_related, args.limit_concepts)
    problems = src.verify()
    if problems:
        print("UMLS files not usable:\n  " + "\n  ".join(problems))
        return 2
    print(f"UMLS release {src.info.source_version}: files found ({', '.join(k for k, v in src.files.items() if v)})")
    if args.check:
        return 0
    if args.stats:
        stats(src)
        return 0
    from src.medical_knowledge.ingest.builder import KnowledgeBuilder
    from src.medical_knowledge.sources.curated import CuratedSource
    from src.medical_knowledge.sources.fixture import FixtureSource

    started = time.time()
    builder = KnowledgeBuilder(args.output, "full")
    try:
        for source in (CuratedSource(), FixtureSource(), src):
            s = source.export(builder)
            print(f"  {s.source_id}: {s.concepts:,} concepts, {s.terms:,} terms, {s.relations:,} relations, {s.abbreviations:,} abbreviations")
        manifest = builder.finalize()
    except Exception:
        builder.abort()
        raise
    c = manifest["counts"]
    print(f"{manifest['name']} {manifest['build_version']} [full]: {c['concepts']:,} concepts, {c['terms']:,} terms, {c['relationships']:,} relationships, "
          f"{manifest['size_bytes'] / 1e6:.0f} MB, {time.time() - started:.0f} s; UMLS concepts skipped for their semantic type: {src.skipped_types:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
