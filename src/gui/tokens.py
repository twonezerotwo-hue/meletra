"""Design tokens of MELETRA: one place for colours, spacing, radii, type sizes and control heights, so that no widget invents its own values.

The palette comes from the logo (deep navy, teal, muted gold). Teal is the interaction colour (and the focus ring); gold is a small accent (wordmark, selected details, the ring on teal-filled buttons); the status
colours are controlled, never neon. ``dark`` is the brand look; ``light`` uses the same roles for people who choose it in Settings.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------------------------------------------------------------- scale
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32, "huge": 48}
RADIUS = {"small": 6, "control": 8, "card": 12, "drop": 16}
FONT_FAMILY = ('"Segoe UI Variable Text", "Segoe UI", ".AppleSystemUIFont", "SF Pro Text", "Helvetica Neue", "Inter", '
               '"Malgun Gothic", "Nirmala UI", "Microsoft YaHei UI", "Apple SD Gothic Neo", "Kohinoor Devanagari", "PingFang SC", sans-serif')      # Latin first; Hangul, Devanagari and Han after it
FONT_PX = {"display": 34, "page": 28, "title": 24, "lead": 20, "section": 17, "name": 16, "reading": 15, "body": 14, "secondary": 13, "meta": 12, "tiny": 11,
           "button": 14, "cta": 15}      # one scale: nothing in the program uses a size that is not in it
WEIGHT = {"regular": 400, "medium": 500, "semibold": 600}
HEIGHT = {"control": 40, "button": 40, "small_button": 32, "cta": 52, "chip": 22, "nav": 40, "card_min": 84}
ELEVATION = {"none": "none", "raised": "0 1px 2px rgba(0,0,0,0.35)"}                          # documentation only: Qt style sheets have no box shadows
TRANSITION_MS = 150                                                                              # hover / drag-over fades (120-180 ms)
FOCUS_WIDTH = 2


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str                 # window
    sidebar: str
    surface: str            # cards, panels
    surface_hover: str      # raised tone: hover, selected rows, drag-over
    field: str              # inputs
    border: str
    border_strong: str
    text: str               # warm white
    text2: str              # soft blue-grey
    text3: str              # metadata (still readable)
    teal: str               # primary interaction colour
    teal_hover: str
    teal_press: str
    on_teal: str
    teal_soft: str          # tinted background for teal chips / selected nav
    gold: str               # accent; the focus ring on teal-filled buttons
    focus: str              # the keyboard-focus ring everywhere else (teal)
    gold_soft: str
    success: str
    success_bg: str
    warning: str
    warning_bg: str
    error: str
    error_bg: str
    neutral: str            # queued
    neutral_bg: str
    scrim: str              # the dimming layer while dragging


DARK = Palette(
    name="dark", bg="#08121F", sidebar="#0B1727", surface="#0F1C2F", surface_hover="#16263D", field="#0B1626",
    border="#1F3049", border_strong="#2E4465", text="#F2EEE4", text2="#A9B8CE", text3="#8496B0",
    teal="#2BC4C4", teal_hover="#4AD8D8", teal_press="#1FA8AB", on_teal="#04141C", teal_soft="#0F3339",
    gold="#E0B04A", gold_soft="#3A2F14", focus="#4AD8D8",
    success="#58CC92", success_bg="#11372B", warning="#EDB254", warning_bg="#3C2E12", error="#EF8A92", error_bg="#3F1D26",
    neutral="#B3C2D8", neutral_bg="#1B2B44", scrim="rgba(4, 10, 18, 150)")

LIGHT = Palette(
    name="light", bg="#F3F6F9", sidebar="#E9EFF5", surface="#FFFFFF", surface_hover="#EEF3F8", field="#FFFFFF",
    border="#D3DCE7", border_strong="#B7C4D4", text="#0E1B2B", text2="#44566D", text3="#566880",
    teal="#08797E", teal_hover="#066A6E", teal_press="#055A5D", on_teal="#FFFFFF", teal_soft="#E3F4F4",
    gold="#9A6F12", gold_soft="#F6EBD0", focus="#066A6E",
    success="#17724A", success_bg="#E1F3EA", warning="#8C5A0E", warning_bg="#FBEFD6", error="#B0303A", error_bg="#FBE4E6",
    neutral="#41546C", neutral_bg="#E3EAF2", scrim="rgba(240, 244, 248, 170)")

PALETTES = {"dark": DARK, "light": LIGHT}
_current = DARK


def set_current(name: str) -> Palette:
    global _current
    _current = PALETTES.get(name, DARK)
    return _current


def current() -> Palette:
    return _current


def color(role: str) -> str:
    """The colour of a palette role (``teal``, ``text2`` ...) in the theme that is active now."""
    return getattr(_current, role)


# --------------------------------------------------------------------------------------------------------------------------------- contrast
def _channel(value: int) -> float:
    v = value / 255.0
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(foreground: str, background: str) -> float:
    """WCAG contrast ratio of two ``#rrggbb`` colours (4.5 = AA for normal text, 3.0 = AA for large text and UI components)."""
    a, b = luminance(foreground), luminance(background)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)
