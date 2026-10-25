"""Production brand assets of MELETRA (Medical Learning & Transcription), made from the existing symbol (assets/brand/meletra_symbol_source.png).

The SYMBOL (arch, crescent, book, caduceus) is the one supplied with the product; it is only cropped, scaled and separated from its flat dark-blue background - it is NOT redrawn.
The old lettering of the supplied image is NOT used: the MELETRA wordmark and the tagline are set here in Windows' Constantia (wordmark, gold gradient) and Segoe UI (tagline, teal).

    python tools/make_brand_assets.py [--source PATH]

Written to assets/brand/:
    meletra_logo.png             the whole lock-up (symbol, MELETRA, tagline), transparent background, 800 x 900
    meletra_logo_horizontal.png  the symbol on the left, MELETRA and the tagline on the right, transparent background, 480 px high
    meletra_symbol.png           the SYMBOL only, transparent background, 256 x 256 (sidebar, About, the drop overlay)
    meletra_icon.png             the application icon master: the symbol on the brand background in a rounded square, 1024 x 1024
    meletra_icon_<n>.png         the same icon at 16 24 32 48 64 128 256 512 1024 px (small sizes: the symbol only, never the lettering)
    meletra.ico                  Windows icon (16 24 32 48 64 128 256), PNG entries
    meletra.icns                 macOS icon (16 32 64 128 256 512 1024)
    installer/wizard_*.bmp       the installer's welcome panel (symbol + MELETRA + tagline) and header picture at 100 / 150 / 200 %
    favicon.ico, favicon-32.png, favicon-192.png, apple-touch-icon.png      for the download page
    brand_assets.json            sizes and SHA-256 of every file
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_tools.icns import ICNS_SIZES, icns_bytes  # noqa: E402
from build_tools.icon import ICON_SIZES, ico_bytes  # noqa: E402

BRAND = ROOT / "assets" / "brand"
SOURCE = BRAND / "meletra_symbol_source.png"
PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
SYMBOL_FRACTION = 0.70                      # rows above this belong to the symbol (measured on the supplied image)
SMALL_MARGIN = 0.04                         # border around the symbol in the 16-48 px icons
MARGIN = 0.12                               # border around the symbol in the larger icons, fraction of the icon edge
NAME = "MELETRA"
TAGLINE = "Medical Learning & Transcription"
GOLD_TOP, GOLD_BOTTOM = (246, 222, 148), (201, 150, 46)       # the wordmark: a vertical gold gradient
TEAL = (111, 208, 207)                                        # the tagline
FONT_WORD = ("constanb.ttf", "constan.ttf", "georgiab.ttf", "timesbd.ttf")
FONT_TAG = ("segoeui.ttf", "seguisb.ttf", "arial.ttf")


def load(source: Path) -> Image.Image:
    return Image.open(source).convert("RGB")


def background_rows(rgb: np.ndarray) -> np.ndarray:
    """The flat vertical gradient behind the symbol: one colour per row (median of the 24 left-most pixels, which hold no artwork)."""
    return np.median(rgb[:, :24, :], axis=1)


def symbol_box(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the symbol: everything clearly brighter than the background in the upper part of the image."""
    bg = background_rows(rgb)
    h = rgb.shape[0]
    diff = np.abs(rgb.astype(int) - bg[:, None, :].astype(int)).sum(axis=2)
    ys, xs = np.where(diff[: int(h * SYMBOL_FRACTION)] > 90)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def transparent(rgb: np.ndarray) -> Image.Image:
    """The image without its background: alpha from the distance to the background gradient, colours un-mixed from it."""
    bg = background_rows(rgb).astype(float)[:, None, :]
    px = rgb.astype(float)
    dist = np.abs(px - bg).max(axis=2)
    alpha = np.clip((dist - 10.0) / 70.0, 0.0, 1.0)
    safe = np.maximum(alpha, 1e-3)[:, :, None]
    color = np.clip((px - bg * (1.0 - alpha[:, :, None])) / safe, 0, 255)
    out = np.dstack([color, alpha * 255.0]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def symbol_only(full: Image.Image, rgb: np.ndarray) -> Image.Image:
    """The symbol alone on a transparent background (the lettering of the supplied image is cut away)."""
    return full.crop(symbol_box(rgb))


def square(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = side / 2
    return image.crop((round(cx - half), round(cy - half), round(cx + half), round(cy + half)))


def font(candidates: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    for name in candidates:
        for folder in (Path("C:/Windows/Fonts"), Path("/Library/Fonts"), Path("/usr/share/fonts/truetype")):
            if (folder / name).is_file():
                return ImageFont.truetype(str(folder / name), size)
    raise SystemExit(f"none of the fonts {candidates} is installed (the wordmark is set in a Windows system font)")


def wordmark(height: int) -> Image.Image:
    """MELETRA in gold, letter-spaced capitals, ``height`` px tall (cap height), transparent background."""
    face = font(FONT_WORD, round(height * 1.38))
    tracking = round(height * 0.16)
    widths = [face.getlength(ch) for ch in NAME]
    total = round(sum(widths) + tracking * (len(NAME) - 1))
    pad = round(height * 0.2)
    mask = Image.new("L", (total + 2 * pad, round(height * 2.0)), 0)
    draw = ImageDraw.Draw(mask)
    x = float(pad)
    for ch, w in zip(NAME, widths):
        draw.text((x, 0), ch, font=face, fill=255)
        x += w + tracking
    box = mask.getbbox()
    mask = mask.crop((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2))
    gradient = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    ramp = np.linspace(0, 1, mask.height)[:, None]
    for i in range(3):
        gradient[:, :, i] = (GOLD_TOP[i] * (1 - ramp) + GOLD_BOTTOM[i] * ramp).astype(np.uint8)
    gradient[:, :, 3] = np.asarray(mask)
    return Image.fromarray(gradient, "RGBA")


def tagline(width: int) -> Image.Image:
    """The tagline in teal, scaled to ``width`` px wide, transparent background."""
    face = font(FONT_TAG, 96)
    pad = 6
    probe = Image.new("L", (int(face.getlength(TAGLINE)) + 2 * pad, 160), 0)
    ImageDraw.Draw(probe).text((pad, 0), TAGLINE, font=face, fill=255)
    box = probe.getbbox()
    probe = probe.crop(box)
    layer = np.zeros((probe.height, probe.width, 4), dtype=np.uint8)
    layer[:, :, 0], layer[:, :, 1], layer[:, :, 2] = TEAL
    layer[:, :, 3] = np.asarray(probe)
    image = Image.fromarray(layer, "RGBA")
    return image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)


def lockup(symbol: Image.Image, canvas_width: int = 800, canvas_height: int = 900) -> Image.Image:
    """Symbol above MELETRA above the tagline, centred, on a transparent canvas."""
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    side = round(canvas_height * 0.60)
    sym = symbol.resize((side, round(symbol.height * side / symbol.width)), Image.LANCZOS)
    word = wordmark(round(canvas_height * 0.085))
    word = word.resize((min(word.width, round(canvas_width * 0.86)), round(word.height * min(word.width, round(canvas_width * 0.86)) / word.width)), Image.LANCZOS)
    tag = tagline(round(word.width * 0.92))
    gap1, gap2 = round(canvas_height * 0.035), round(canvas_height * 0.03)
    block = sym.height + gap1 + word.height + gap2 + tag.height
    top = (canvas_height - block) // 2
    canvas.alpha_composite(sym, ((canvas_width - sym.width) // 2, top))
    canvas.alpha_composite(word, ((canvas_width - word.width) // 2, top + sym.height + gap1))
    canvas.alpha_composite(tag, ((canvas_width - tag.width) // 2, top + sym.height + gap1 + word.height + gap2))
    return canvas


def horizontal_logo(symbol: Image.Image, height: int = 480) -> Image.Image:
    """The symbol on the left, MELETRA and the tagline stacked on the right: for wide places such as a page header."""
    sym = symbol.resize((round(symbol.width * height / symbol.height), height), Image.LANCZOS)
    word = wordmark(round(height * 0.22))
    tag = tagline(word.width)
    gap = round(height * 0.05)
    text_height = word.height + gap + tag.height
    canvas = Image.new("RGBA", (sym.width + round(height * 0.10) + word.width, height), (0, 0, 0, 0))
    canvas.alpha_composite(sym, (0, 0))
    left = sym.width + round(height * 0.10)
    top = (height - text_height) // 2
    canvas.alpha_composite(word, (left, top))
    canvas.alpha_composite(tag, (left, top + word.height + gap))
    return canvas


def icon_master(rgb: np.ndarray, size: int = 1024, margin: float = MARGIN) -> Image.Image:
    """The symbol on the brand background, inside a rounded square (transparent corners), with an empty border around the symbol."""
    bg = background_rows(rgb)
    top, bottom = bg[0], bg[-1]
    ramp = np.linspace(0, 1, size)[:, None, None]
    canvas = (top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp) * np.ones((1, size, 1))
    base = Image.fromarray(canvas.astype(np.uint8), "RGB").convert("RGBA")
    left, top_, right, bottom_ = symbol_box(rgb)
    symbol = square(transparent(rgb), (left - 12, top_ - 12, right + 12, bottom_ + 12))
    inner = round(size * (1 - 2 * margin))
    symbol = symbol.resize((inner, inner), Image.LANCZOS)
    base.alpha_composite(symbol, ((size - inner) // 2, (size - inner) // 2))
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), radius=round(size * 4 * 0.22), fill=255)
    base.putalpha(mask.resize((size, size), Image.LANCZOS))
    return base


def gradient_canvas(rgb: np.ndarray, width: int, height: int) -> Image.Image:
    """The brand's flat dark-blue background (the vertical gradient measured on the supplied image) at any size."""
    bg = background_rows(rgb)
    ramp = np.linspace(0, 1, height)[:, None, None]
    canvas = (bg[0][None, None, :] * (1 - ramp) + bg[-1][None, None, :] * ramp) * np.ones((1, width, 1))
    return Image.fromarray(canvas.astype(np.uint8), "RGB").convert("RGBA")


def installer_images(symbol: Image.Image, rgb: np.ndarray) -> dict[str, bytes]:
    """The Windows installer's wizard pictures (Inno Setup, 24-bit BMP, at 100 %, 150 % and 200 % scale): a tall panel with the symbol, MELETRA and the tagline, and a small square symbol."""
    files: dict[str, bytes] = {}

    def bmp(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "BMP")
        return buffer.getvalue()

    for scale, tag in ((1.0, "100"), (1.5, "150"), (2.0, "200")):
        w, h = round(164 * scale), round(314 * scale)
        panel = gradient_canvas(rgb, w, h)
        side = round(w * 0.78)
        sym = symbol.resize((side, round(symbol.height * side / symbol.width)), Image.LANCZOS)
        word = wordmark(round(w * 0.105))
        word = word.resize((round(w * 0.88), round(word.height * round(w * 0.88) / word.width)), Image.LANCZOS)
        line = tagline(round(w * 0.80))
        gap1, gap2 = round(h * 0.035), round(h * 0.03)
        top = (h - (sym.height + gap1 + word.height + gap2 + line.height)) // 2
        panel.alpha_composite(sym, ((w - sym.width) // 2, top))
        panel.alpha_composite(word, ((w - word.width) // 2, top + sym.height + gap1))
        panel.alpha_composite(line, ((w - line.width) // 2, top + sym.height + gap1 + word.height + gap2))
        files[f"installer/wizard_large_{tag}.bmp"] = bmp(panel)
        small = round(55 * scale)
        tile = gradient_canvas(rgb, small, small)
        inner = round(small * 0.92)
        mark = symbol.resize((inner, round(symbol.height * inner / symbol.width)), Image.LANCZOS)
        tile.alpha_composite(mark, ((small - mark.width) // 2, (small - mark.height) // 2))
        files[f"installer/wizard_small_{tag}.bmp"] = bmp(tile)
    return files


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def make(source: Path = SOURCE, out: Path = BRAND) -> dict:
    out = Path(out)
    (out / "installer").mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(load(source))
    full = transparent(rgb)
    symbol = symbol_only(full, rgb)
    files: dict[str, bytes] = {}
    box = symbol_box(rgb)
    square_symbol = square(full, (box[0] - 12, box[1] - 12, box[2] + 12, box[3] + 12))
    files["meletra_symbol.png"] = png_bytes(square_symbol.resize((256, 256), Image.LANCZOS))
    files["meletra_logo.png"] = png_bytes(lockup(square_symbol))
    files["meletra_logo_horizontal.png"] = png_bytes(horizontal_logo(square_symbol))
    files.update(installer_images(square_symbol, rgb))
    master = icon_master(rgb)
    small = icon_master(rgb, 256, SMALL_MARGIN)                                      # 16-48 px: the symbol fills more of the square so it stays readable
    files["meletra_icon.png"] = png_bytes(master)
    pngs = {n: png_bytes((small if n <= 48 else master).resize((n, n), Image.LANCZOS)) for n in PNG_SIZES}
    for n, data in pngs.items():
        files[f"meletra_icon_{n}.png"] = data
    files["meletra.ico"] = ico_bytes({n: pngs[n] for n in ICON_SIZES})
    files["meletra.icns"] = icns_bytes({n: pngs[n] for n in ICNS_SIZES})
    files["favicon.ico"] = ico_bytes({n: pngs[n] for n in (16, 32, 48)})
    files["favicon-32.png"] = pngs[32]
    files["favicon-192.png"] = png_bytes(master.resize((192, 192), Image.LANCZOS))
    files["apple-touch-icon.png"] = png_bytes(master.resize((180, 180), Image.LANCZOS))
    manifest = {}
    for name, data in sorted(files.items()):
        (out / name).write_bytes(data)
        manifest[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["_source"] = {"file": source.name, "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(), "symbol_box": list(box),
                           "wordmark_font": "Constantia Bold (Windows system font)", "tagline_font": "Segoe UI (Windows system font)"}
    (out / "brand_assets.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    a = ap.parse_args(argv)
    manifest = make(a.source)
    print(f"{len(manifest) - 1} brand files written to {BRAND}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
