"""The LGPL media swap (build_tools/lgpl_media.py): PyAV's GPL-built FFmpeg is replaced by an LGPL-only build of the same release, and nothing GPL-class is left."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from build_tools import lgpl_media as L

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(sys.prefix) / "Lib" / "site-packages"
WINDOWS_PYAV = (SITE / "av").is_dir() and (SITE / "av.libs").is_dir()
PINNED_ZIP = ROOT / "build_cache" / "ffmpeg-lgpl" / L.PINNED["name"]


def test_names_and_flags():
    assert L.plain_name("avcodec-62-984de33114b7fa384296817dec999c9d.dll") == "avcodec-62.dll"
    assert L.plain_name("swscale-9-0c9886c118598c54e159ef2267ff99f3.dll") == "swscale-9.dll"
    assert L.plain_name("libx264-165-f3a909470ddc2d85ed21eda3d0fb7954.dll") == ""              # not an FFmpeg library: removed, never mapped
    assert L.plain_name("avcodec-62.dll") == ""
    assert L.gpl_flags(b"... --enable-gpl --enable-libx264 ...") == ["--enable-gpl", "--enable-libx264"]
    assert L.gpl_flags(b"... --disable-gpl --enable-version3 --enable-libdav1d ...") == []      # an LGPL build says --disable-gpl, which is not --enable-gpl


def fake_zip(path: Path, config: bytes = b"--disable-gpl --enable-version3", skip: str = "") -> Path:
    names = {"avcodec": 62, "avdevice": 62, "avfilter": 11, "avformat": 62, "avutil": 60, "swresample": 6, "swscale": 9}
    with zipfile.ZipFile(path, "w") as z:
        for lib, major in names.items():
            if lib != skip:
                z.writestr(f"ffmpeg-x-lgpl-shared/bin/{lib}-{major}.dll", b"MZ" + config)
        z.writestr("ffmpeg-x-lgpl-shared/LICENSE.txt", "GNU LESSER GENERAL PUBLIC LICENSE")
    return path


def test_the_archive_must_be_complete_and_lgpl(tmp_path):
    good = L.lgpl_dlls_from_zip(fake_zip(tmp_path / "good.zip"))
    assert sorted(good) == ["avcodec-62.dll", "avdevice-62.dll", "avfilter-11.dll", "avformat-62.dll", "avutil-60.dll", "swresample-6.dll", "swscale-9.dll"]
    with pytest.raises(ValueError, match="GPL-class"):
        L.lgpl_dlls_from_zip(fake_zip(tmp_path / "gpl.zip", b"--enable-gpl --enable-libx264"))
    with pytest.raises(FileNotFoundError, match="swscale"):
        L.lgpl_dlls_from_zip(fake_zip(tmp_path / "short.zip", skip="swscale"))
    assert "LESSER" in L.license_text_from_zip(tmp_path / "good.zip")


def test_a_gpl_or_x264_file_is_found(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "avcodec-62.dll").write_bytes(b"MZ --enable-gpl")
    (tmp_path / "libx265-abc.dll").write_bytes(b"MZ")
    (tmp_path / "clean.dll").write_bytes(b"MZ --disable-gpl")
    assert L.gpl_class_files(tmp_path) == ["a\\avcodec-62.dll" if sys.platform == "win32" else "a/avcodec-62.dll", "libx265-abc.dll"]


@pytest.mark.skipif(not WINDOWS_PYAV, reason="needs the Windows PyAV wheel (av + av.libs)")
def test_the_import_table_of_a_pyav_module_is_rewritten_in_place(tmp_path):
    source = SITE / "av" / "_core.pyd"
    target = tmp_path / "_core.pyd"
    shutil.copy2(source, target)
    before = L.imported_dlls(target)
    mangled = [n for n in before if L.MANGLED.match(n)]
    assert mangled
    changes = L.patch_imports(target, {n: L.plain_name(n) for n in mangled})
    after = L.imported_dlls(target)
    assert len(changes) == len(mangled) and target.stat().st_size == source.stat().st_size
    assert not any(L.MANGLED.match(n) for n in after) and all(L.plain_name(n) in after for n in mangled)
    with pytest.raises(ValueError, match="longer"):
        L.patch_imports(target, {after[0]: after[0] + "-much-longer-name.dll"})


@pytest.mark.skipif(not (WINDOWS_PYAV and PINNED_ZIP.is_file()), reason="needs the Windows PyAV wheel and the pinned LGPL archive in build_cache/ffmpeg-lgpl")
def test_the_swapped_pyav_decodes_and_has_no_gpl_encoder(tmp_path):
    copy = tmp_path / "site"
    copy.mkdir()
    shutil.copytree(SITE / "av", copy / "av")
    shutil.copytree(SITE / "av.libs", copy / "av.libs")
    report = L.swap_pyav(copy / "av", copy / "av.libs", L.pinned_archive(PINNED_ZIP.parent))
    assert report["installed"] and any(n.startswith("libx264") for n in report["removed"])
    assert L.gpl_class_files(copy) == []
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "tone.wav"
    sf.write(str(wav), (0.2 * np.sin(np.arange(16000) / 5.0)).astype("float32"), 16000)
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import av\n"
              "assert 'site' in av.__file__, av.__file__\n"
              "from faster_whisper.audio import decode_audio\n"
              "a = decode_audio(sys.argv[2], sampling_rate=16000); assert abs(len(a) - 16000) < 400, len(a)\n"
              "import av.codec\n"
              "for enc in ('libx264', 'libx265'):\n"
              "    try:\n        av.codec.Codec(enc, 'w')\n    except Exception:\n        continue\n"
              "    raise SystemExit(enc + ' is available')\n"
              "print('OK', av.ffmpeg_version_info)\n")
    done = subprocess.run([sys.executable, "-c", script, str(copy), str(wav)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and "OK" in done.stdout, done.stdout + done.stderr
