#!/bin/bash
# Build an LGPL-only FFmpeg (shared libraries, no programs) from a pinned source tarball, for the Intel Mac app.
#
#   bash scripts/build_macos_ffmpeg_lgpl.sh VERSION SHA256 PREFIX
#
# The libraries replace the GPL-built FFmpeg inside the PyAV and OpenCV wheels (build_tools/lgpl_media_macos.py): no libx264, no libx265, no external codec library at all
# (--disable-autodetect), no GPL and no non-free code. The program only DECODES media and every decoder it needs is native FFmpeg code.
# STATUS: written without a Mac to run it on; first exercised by the CI run that uses `media=lgpl`.
set -euo pipefail

VERSION="${1:?version, for example 8.1.3}"
SHA256="${2:?sha256 of ffmpeg-VERSION.tar.xz}"
PREFIX="${3:?install prefix}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

curl -fsSL --proto '=https' -o "$WORK/ffmpeg.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz"
echo "${SHA256}  $WORK/ffmpeg.tar.xz" | shasum -a 256 -c -
tar -xf "$WORK/ffmpeg.tar.xz" -C "$WORK"
cd "$WORK/ffmpeg-${VERSION}"

command -v nasm >/dev/null 2>&1 || brew install nasm          # build tool only (BSD); makes the H.264 / HEVC decoders fast

./configure \
  --prefix="$PREFIX" \
  --enable-shared --disable-static \
  --disable-gpl --disable-nonfree --disable-autodetect \
  --enable-zlib \
  --disable-programs --disable-doc --disable-debug \
  --enable-pic \
  --extra-cflags="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}" \
  --extra-ldflags="-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}" 2>&1 | tee "$WORK/configure.log"

grep -q "License: LGPL version 2.1 or later" "$WORK/configure.log" || { echo "::error::FFmpeg ${VERSION} is not configured as LGPL-2.1+ (see configure.log above)"; exit 1; }

make -j"$(sysctl -n hw.ncpu)"
make install

# refuse any GPL-class result before the CI uses it
for lib in "$PREFIX"/lib/lib*.dylib; do
  [ -L "$lib" ] && continue
  if grep -a -q -E -e '--enable-(gpl|nonfree|libx264|libx265)' "$lib"; then echo "::error::$lib carries a GPL-class option"; exit 1; fi
done
if ls "$PREFIX"/lib | grep -q -E 'x26[45]'; then echo "::error::x264 / x265 present in $PREFIX/lib"; exit 1; fi
echo "LGPL-only FFmpeg ${VERSION} installed in ${PREFIX}:"; ls "$PREFIX/lib" | grep -E '^lib(av|sw)[a-z]+\.[0-9]+\.dylib$'
