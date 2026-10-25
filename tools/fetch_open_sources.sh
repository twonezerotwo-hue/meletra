#!/bin/bash
# Download the raw files of the sources of the PUBLIC knowledge pack (redistributable licences only) into data/open_raw/ - used by CI; the files are git-ignored.
#   MeSH (NLM, public domain with attribution), Gene Ontology (CC BY 4.0), HGNC (CC0), SCOWL (permissive, copyright notice shipped with it).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/open_raw && cd data/open_raw
get() { until curl -fL -sS -C - --retry 8 --retry-delay 5 --connect-timeout 30 -o "$1" "$2"; do echo "retry $1"; sleep 5; done; echo "done $1 $(wc -c < "$1")"; }
get desc2026.gz "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"
get hgnc_complete_set.txt "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
get go-basic.obo "https://purl.obolibrary.org/obo/go/go-basic.obo"
if [ ! -d scowl/final ]; then
  get scowl.tar.gz "https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz"
  rm -rf scowl && mkdir scowl && tar -xzf scowl.tar.gz -C scowl --strip-components=1
fi
test -f scowl/final/english-words.10 && echo "SCOWL ready"
echo ALL_DONE
