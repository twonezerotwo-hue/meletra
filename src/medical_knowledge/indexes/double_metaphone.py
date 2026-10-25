"""Double Metaphone (Lawrence Philips), written for this project; no length limit on the codes (the classic 4-letter cut-off is useless for medical terms).

``double_metaphone("Schwann")`` -> ``("XFN", "SFN")``: a primary and an alternate pronunciation key. English-oriented; medical words are mostly
Greek / Latin / eponyms, so it is evaluated against the consonant skeleton and the medical phoneme key (milestone 4) and only used where it helps.
"""

from __future__ import annotations

_VOWELS = "AEIOUY"


def double_metaphone(word: str) -> tuple[str, str]:
    st = "".join(ch for ch in word.upper() if ch.isalpha() or ch == "Ç")
    n = len(st)
    if n == 0:
        return "", ""
    pad = st + "     "
    pri: list[str] = []
    sec: list[str] = []

    def add(p: str, s: str | None = None) -> None:
        pri.append(p)
        sec.append(p if s is None else s)

    def at(start: int, length: int, *options: str) -> bool:
        if start < 0 or start >= n:
            return False
        return pad[start:start + length] in options

    slavo = "W" in st or "K" in st or "CZ" in st or "WITZ" in st

    def vowel(i: int) -> bool:
        return 0 <= i < n and st[i] in _VOWELS

    pos = 0
    if at(0, 2, "GN", "KN", "PN", "WR", "PS"):
        pos = 1
    if st[0] == "X":
        add("S")
        pos = 1
    last = n - 1
    while pos < n:
        c = st[pos]
        if c in _VOWELS:
            if pos == 0:
                add("A")
            pos += 1
        elif c == "B":
            add("P")
            pos += 2 if pad[pos + 1] == "B" else 1
        elif c == "Ç":
            add("S")
            pos += 1
        elif c == "C":
            if pos > 1 and not vowel(pos - 2) and at(pos - 1, 3, "ACH") and pad[pos + 2] != "I" and (pad[pos + 2] != "E" or at(pos - 2, 6, "BACHER", "MACHER")):
                add("K")
                pos += 2
            elif pos == 0 and at(pos, 6, "CAESAR"):
                add("S")
                pos += 2
            elif at(pos, 4, "CHIA"):
                add("K")
                pos += 2
            elif at(pos, 2, "CH"):
                if pos > 0 and at(pos, 4, "CHAE"):
                    add("K", "X")
                elif pos == 0 and (at(pos + 1, 5, "HARAC", "HARIS") or at(pos + 1, 3, "HOR", "HYM", "HIA", "HEM")) and not at(0, 5, "CHORE"):
                    add("K")
                elif at(0, 4, "VAN ", "VON ") or at(0, 3, "SCH") or at(pos - 2, 6, "ORCHES", "ARCHIT", "ORCHID") or at(pos + 2, 1, "T", "S") or \
                        ((at(pos - 1, 1, "A", "O", "U", "E") or pos == 0) and at(pos + 2, 1, "L", "R", "N", "M", "B", "H", "F", "V", "W", " ")):
                    add("K")
                elif pos > 0:
                    add("K") if at(0, 2, "MC") else add("X", "K")
                else:
                    add("X")
                pos += 2
            elif at(pos, 2, "CZ") and not at(pos - 2, 4, "WICZ"):
                add("S", "X")
                pos += 2
            elif at(pos + 1, 3, "CIA"):
                add("X")
                pos += 3
            elif at(pos, 2, "CC") and not (pos == 1 and st[0] == "M"):
                if at(pos + 2, 1, "I", "E", "H") and not at(pos + 2, 2, "HU"):
                    if (pos == 1 and st[0] == "A") or at(pos - 1, 5, "UCCEE", "UCCES"):
                        add("KS")
                    else:
                        add("X")
                    pos += 3
                else:
                    add("K")
                    pos += 2
            elif at(pos, 2, "CK", "CG", "CQ"):
                add("K")
                pos += 2
            elif at(pos, 2, "CI", "CE", "CY"):
                if at(pos, 3, "CIO", "CIE", "CIA"):
                    add("S", "X")
                else:
                    add("S")
                pos += 2
            else:
                add("K")
                if at(pos + 1, 2, " C", " Q", " G"):
                    pos += 3
                elif at(pos + 1, 1, "C", "K", "Q") and not at(pos + 1, 2, "CE", "CI"):
                    pos += 2
                else:
                    pos += 1
        elif c == "D":
            if at(pos, 2, "DG"):
                if at(pos + 2, 1, "I", "E", "Y"):
                    add("J")
                    pos += 3
                else:
                    add("TK")
                    pos += 2
            elif at(pos, 2, "DT", "DD"):
                add("T")
                pos += 2
            else:
                add("T")
                pos += 1
        elif c == "F":
            add("F")
            pos += 2 if pad[pos + 1] == "F" else 1
        elif c == "G":
            if pad[pos + 1] == "H":
                if pos > 0 and not vowel(pos - 1):
                    add("K")
                    pos += 2
                elif pos == 0:
                    add("J" if pad[pos + 2] == "I" else "K")
                    pos += 2
                elif (pos > 1 and at(pos - 2, 1, "B", "H", "D")) or (pos > 2 and at(pos - 3, 1, "B", "H", "D")) or (pos > 3 and at(pos - 4, 1, "B", "H")):
                    pos += 2
                else:
                    if pos > 2 and pad[pos - 1] == "U" and at(pos - 3, 1, "C", "G", "L", "R", "T"):
                        add("F")
                    elif pos > 0 and pad[pos - 1] != "I":
                        add("K")
                    pos += 2
            elif pad[pos + 1] == "N":
                if pos == 1 and vowel(0) and not slavo:
                    add("KN", "N")
                elif not at(pos + 2, 2, "EY") and pad[pos + 1] != "Y" and not slavo:
                    add("N", "KN")
                else:
                    add("KN")
                pos += 2
            elif at(pos + 1, 2, "LI") and not slavo:
                add("KL", "L")
                pos += 2
            elif pos == 0 and (pad[pos + 1] == "Y" or at(pos + 1, 2, "ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE", "EI", "ER")):
                add("K", "J")
                pos += 2
            elif (at(pos + 1, 2, "ER") or pad[pos + 1] == "Y") and not at(0, 6, "DANGER", "RANGER", "MANGER") and not at(pos - 1, 1, "E", "I") and not at(pos - 1, 3, "RGY", "OGY"):
                add("K", "J")
                pos += 2
            elif at(pos + 1, 1, "E", "I", "Y") or at(pos - 1, 4, "AGGI", "OGGI"):
                if at(0, 4, "VAN ", "VON ") or at(0, 3, "SCH") or at(pos + 1, 2, "ET"):
                    add("K")
                elif at(pos + 1, 4, "IER "):
                    add("J")
                else:
                    add("J", "K")
                pos += 2
            else:
                add("K")
                pos += 2 if pad[pos + 1] == "G" else 1
        elif c == "H":
            if (pos == 0 or vowel(pos - 1)) and vowel(pos + 1):
                add("H")
                pos += 2
            else:
                pos += 1
        elif c == "J":
            if at(pos, 4, "JOSE") or at(0, 4, "SAN "):
                if (pos == 0 and pad[pos + 4] == " ") or at(0, 4, "SAN "):
                    add("H")
                else:
                    add("J", "H")
                pos += 1
            else:
                if pos == 0 and not at(pos, 4, "JOSE"):
                    add("J", "A")
                elif vowel(pos - 1) and not slavo and at(pos + 1, 1, "A", "O"):
                    add("J", "H")
                elif pos == last:
                    add("J", " ")
                elif not at(pos + 1, 1, "L", "T", "K", "S", "N", "M", "B", "Z") and not at(pos - 1, 1, "S", "K", "L"):
                    add("J")
                pos += 2 if pad[pos + 1] == "J" else 1
        elif c == "K":
            add("K")
            pos += 2 if pad[pos + 1] == "K" else 1
        elif c == "L":
            if pad[pos + 1] == "L":
                if (pos == n - 3 and at(pos - 1, 4, "ILLO", "ILLA", "ALLE")) or ((at(last - 1, 2, "AS", "OS") or at(last, 1, "A", "O")) and at(pos - 1, 4, "ALLE")):
                    add("L", " ")
                    pos += 2
                    continue
                pos += 2
            else:
                pos += 1
            add("L")
        elif c == "M":
            add("M")
            if (at(pos - 1, 3, "UMB") and (pos + 1 == last or at(pos + 2, 2, "ER"))) or pad[pos + 1] == "M":
                pos += 2
            else:
                pos += 1
        elif c in ("N", "Ñ"):
            add("N")
            pos += 2 if pad[pos + 1] == "N" else 1
        elif c == "P":
            if pad[pos + 1] == "H":
                add("F")
                pos += 2
            else:
                add("P")
                pos += 2 if at(pos + 1, 1, "P", "B") else 1
        elif c == "Q":
            add("K")
            pos += 2 if pad[pos + 1] == "Q" else 1
        elif c == "R":
            if pos == last and not slavo and at(pos - 2, 2, "IE") and not at(pos - 4, 2, "ME", "MA"):
                add("", "R")
            else:
                add("R")
            pos += 2 if pad[pos + 1] == "R" else 1
        elif c == "S":
            if at(pos - 1, 3, "ISL", "YSL"):
                pos += 1
            elif pos == 0 and at(pos, 5, "SUGAR"):
                add("X", "S")
                pos += 1
            elif at(pos, 2, "SH"):
                add("S" if at(pos + 1, 4, "HEIM", "HOEK", "HOLM", "HOLZ") else "X")
                pos += 2
            elif at(pos, 3, "SIO", "SIA") or at(pos, 4, "SIAN"):
                add("S") if slavo else add("S", "X")
                pos += 3
            elif (pos == 0 and at(pos + 1, 1, "M", "N", "L", "W")) or at(pos + 1, 1, "Z"):
                add("S", "X")
                pos += 2 if at(pos + 1, 1, "Z") else 1
            elif at(pos, 2, "SC"):
                if pad[pos + 2] == "H":
                    if at(pos + 3, 2, "OO", "ER", "EN", "UY", "ED", "EM"):
                        add("X", "SK") if at(pos + 3, 2, "ER", "EN") else add("SK")
                    elif pos == 0 and not vowel(3) and pad[3] != "W":
                        add("X", "S")
                    else:
                        add("X")
                elif at(pos + 2, 1, "I", "E", "Y"):
                    add("S")
                else:
                    add("SK")
                pos += 3
            else:
                if pos == last and at(pos - 2, 2, "AI", "OI"):
                    add("", "S")
                else:
                    add("S")
                pos += 2 if at(pos + 1, 1, "S", "Z") else 1
        elif c == "T":
            if at(pos, 4, "TION") or at(pos, 3, "TIA", "TCH"):
                add("X")
                pos += 3
            elif at(pos, 2, "TH") or at(pos, 3, "TTH"):
                if at(pos + 2, 2, "OM", "AM") or at(0, 4, "VAN ", "VON ") or at(0, 3, "SCH"):
                    add("T")
                else:
                    add("0", "T")
                pos += 2
            else:
                add("T")
                pos += 2 if at(pos + 1, 1, "T", "D") else 1
        elif c == "V":
            add("F")
            pos += 2 if pad[pos + 1] == "V" else 1
        elif c == "W":
            if at(pos, 2, "WR"):
                add("R")
                pos += 2
            else:
                if pos == 0 and (vowel(pos + 1) or at(pos, 2, "WH")):
                    add("A", "F") if vowel(pos + 1) else add("A")
                if (pos == last and vowel(pos - 1)) or at(pos - 1, 5, "EWSKI", "EWSKY", "OWSKI", "OWSKY") or at(0, 3, "SCH"):
                    add("", "F")
                    pos += 1
                elif at(pos, 4, "WICZ", "WITZ"):
                    add("TS", "FX")
                    pos += 4
                else:
                    pos += 1
        elif c == "X":
            if not (pos == last and (at(pos - 3, 3, "IAU", "EAU") or at(pos - 2, 2, "AU", "OU"))):
                add("KS")
            pos += 2 if at(pos + 1, 1, "C", "X") else 1
        elif c == "Z":
            if pad[pos + 1] == "H":
                add("J")
                pos += 2
                continue
            if at(pos + 1, 2, "ZO", "ZI", "ZA") or (slavo and pos > 0 and pad[pos - 1] != "T"):
                add("S", "TS")
            else:
                add("S")
            pos += 2 if pad[pos + 1] == "Z" else 1
        else:
            pos += 1
    return "".join(pri).strip(), "".join(sec).strip()


def dm_key(text: str) -> str:
    """One key for a phrase: primary codes of its words, concatenated (the compact form: ASR joins and splits compounds at will)."""
    parts = [double_metaphone(w)[0] for w in text.replace("-", " ").split() if w]
    return "".join(parts)
