"""A tiny SYNTHETIC UMLS release in the real RRF format (invented CUIs; not UMLS data) for the importer tests, plus a bulk generator for throughput tests."""

from __future__ import annotations

import random
from pathlib import Path


def conso(cui, lat, ts, stt, ispref, sab, tty, code, text, srl="0", suppress="N", aui=None, scui=""):
    aui = aui or f"A{abs(hash((cui, text, sab, tty))) % 10**7:07d}"
    return "|".join([cui, lat, ts, "L0", stt, "S0", ispref, aui, "", scui, "", sab, tty, code, text, srl, suppress, ""]) + "|"


def write_mini(root: Path) -> Path:
    meta = root / "2099AA" / "META"
    meta.mkdir(parents=True)
    atoms = [
        conso("C9000001", "ENG", "P", "PF", "Y", "MSH", "MH", "D000001", "Nissl Bodies", scui="D000001"),
        conso("C9000001", "ENG", "S", "VO", "N", "MSH", "ET", "D000001", "Nissl Substance", scui="D000001"),
        conso("C9000001", "ENG", "S", "PF", "N", "SNOMEDCT_US", "SY", "1234001", "Tigroid substance", "3"),
        conso("C9000001", "GER", "S", "PF", "N", "MSHGER", "SY", "D000001", "Nissl-Substanz"),
        conso("C9000001", "ENG", "S", "PF", "N", "MSH", "ET", "D000001", "Obsolete Nissl Name", suppress="O"),
        conso("C9000002", "ENG", "P", "PF", "Y", "MSH", "MH", "D000002", "Schwann Cells"),
        conso("C9000002", "ENG", "S", "PF", "N", "MSH", "ET", "D000002", "Neurolemmocytes"),
        conso("C9000002", "ENG", "S", "PF", "N", "MSH", "ET", "D000002", "Hidden Editor Term", suppress="Y"),
        conso("C9000003", "ENG", "P", "PF", "Y", "MSH", "MH", "D000003", "Peripheral Nervous System"),
        conso("C9000003", "ENG", "S", "PF", "N", "MSH", "AB", "D000003", "PNS"),
        conso("C9000003", "LAT", "S", "PF", "N", "MSHLAT", "SY", "D000003", "Systema nervosum periphericum"),
        conso("C9000004", "ENG", "P", "PF", "Y", "MSH", "MH", "D000004", "Tanycytes"),
        conso("C9000005", "ENG", "P", "PF", "Y", "MSH", "MH", "D000005", "Between"),                    # spatial concept: skipped
        conso("C9000006", "ENG", "P", "PF", "Y", "MSH", "MH", "D000006", "Signal Transduction"),
        conso("C9000007", "SPA", "P", "PF", "Y", "MSHSPA", "MH", "D000007", "Cuerpos de Nissl"),         # Spanish: not imported by default
    ]
    (meta / "MRCONSO.RRF").write_text("\n".join(atoms) + "\n", encoding="utf-8")
    sty = [("C9000001", "T026", "A1.2.3.2", "Cell Component"), ("C9000002", "T025", "A1.2.3.1", "Cell"), ("C9000003", "T022", "A1.2.2", "Body System"),
           ("C9000004", "T025", "A1.2.3.1", "Cell"), ("C9000005", "T082", "A2.1.4", "Spatial Concept"), ("C9000006", "T044", "B2.2.1.2", "Molecular Function"),
           ("C9000007", "T026", "A1.2.3.2", "Cell Component")]
    (meta / "MRSTY.RRF").write_text("\n".join("|".join([c, t, n, s, "AT1", ""]) + "|" for c, t, n, s in sty) + "\n", encoding="utf-8")

    def rel(c1, r, c2, rela="", sab="MSH", suppress="N"):
        return "|".join([c1, "A1", "CUI", r, c2, "A2", "CUI", rela, "R1", "", sab, "SL", "", "", suppress, ""]) + "|"

    rels = [rel("C9000002", "PAR", "C9000003", "part_of"), rel("C9000004", "PAR", "C9000002", "isa"), rel("C9000001", "RO", "C9000002"), rel("C9000002", "RO", "C9000001"),
            rel("C9000001", "RB", "C9000003"), rel("C9000001", "CHD", "C9000002"), rel("C9000001", "RO", "C9000099"),            # unknown CUI: dropped
            rel("C9000004", "RO", "C9000006", suppress="O")]
    (meta / "MRREL.RRF").write_text("\n".join(rels) + "\n", encoding="utf-8")
    sabs = []
    for sab, name, ver, srl in (("MSH", "Medical Subject Headings", "2099_2099", "0"), ("MSHGER", "MeSH German", "2099", "3"), ("MSHLAT", "MeSH Latin", "2099", "3"),
                                ("SNOMEDCT_US", "SNOMED CT, US Edition", "2099_01", "3"), ("MSHSPA", "MeSH Spanish", "2099", "3")):
        fields = ["C0", "C0", sab, sab, name, "", ver, "", "", "", "", "", "", srl, "", "", "", "", "", "", "", "", "", "", ""]
        sabs.append("|".join(fields) + "|")
    (meta / "MRSAB.RRF").write_text("\n".join(sabs) + "\n", encoding="utf-8")
    return meta


def write_bulk(root: Path, concepts: int, seed: int = 1) -> Path:
    """``concepts`` synthetic concepts (3 atoms each) in correct CUI order - for throughput / memory tests only."""
    random.seed(seed)
    meta = root / "2099BB" / "META"
    meta.mkdir(parents=True)
    syl = ["ba", "ce", "di", "fo", "gu", "ha", "ji", "ko", "le", "mu", "ne", "po", "ra", "si", "to", "vu", "xe", "zy", "cyt", "ner", "gly", "myo", "oste"]
    with (meta / "MRCONSO.RRF").open("w", encoding="utf-8") as c, (meta / "MRSTY.RRF").open("w", encoding="utf-8") as s:
        for i in range(concepts):
            cui = f"C{5000000 + i:07d}"
            word = "".join(random.choice(syl) for _ in range(random.randint(3, 5)))
            for k, text in enumerate((word.capitalize(), word + "is", word + " syndrome")):
                c.write(conso(cui, "ENG", "P" if k == 0 else "S", "PF", "Y" if k == 0 else "N", "MSH", "MH" if k == 0 else "SY", f"D{i:06d}", text, aui=f"A{i * 3 + k:08d}") + "\n")
            s.write("|".join([cui, "T047", "B2.2.1.2.1", "Disease or Syndrome", "AT1", ""]) + "|\n")
    return meta
