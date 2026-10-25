#!/usr/bin/env bash
# MELETRA for INTEL Macs (macOS 12 or newer): find the NEWEST Mac release on GitHub, download it, check its SHA-256, install it.
#
#   bash install-macos.sh                       download + install into /Applications
#   bash install-macos.sh --dest ~/Applications install somewhere else
#   bash install-macos.sh --keep                keep the downloaded files (in ~/Downloads/MedicalLectureTranscriber)
#
# The repository is private: set GH_TOKEN to a token that can read it, or run `gh auth login` (GitHub CLI) once. If the repository is made public neither is needed.
# Needs about 8 GB of free disk space while it works. The speech model (Whisper Large-v3) is INSIDE the app: no second download, no internet needed to use it.
# The app is downloaded with curl, so macOS does not add the "downloaded from the internet" flag and it opens without the Gatekeeper warning.
set -euo pipefail

REPO="${MLT_REPO:-twonezerotwo-hue/whispermedical}"
DEST="/Applications"
KEEP=0
TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

[ "$(uname -s)" = "Darwin" ] || { echo "This installer is for macOS." >&2; exit 2; }
if [ "$(uname -m)" != "x86_64" ]; then
    echo "This build is for Intel Macs; this Mac is $(uname -m). It is not installed." >&2
    exit 2
fi
APP_NAME="MELETRA.app"
LEGACY_APP_NAMES=("WhisperMedical.app" "Medical Lecture Transcriber.app")   # the names of earlier releases: an old image still holds one, an old install may still be in Applications
BUNDLE_ID="com.medicallecturetranscriber.app"           # unchanged by the renames: it is how an old copy is recognised as this program
WORK="${HOME}/Downloads/MedicalLectureTranscriber"
mkdir -p "$WORK"
FREE_GB=$(df -g "$WORK" | awk 'NR==2{print $4}')
[ "${FREE_GB:-0}" -ge 8 ] || { echo "Only ${FREE_GB} GB free; about 8 GB are needed." >&2; exit 2; }

TOKEN="${GH_TOKEN:-}"
if [ -z "$TOKEN" ] && command -v gh >/dev/null 2>&1; then TOKEN="$(gh auth token 2>/dev/null || true)"; fi
auth=()
[ -n "$TOKEN" ] && auth=(-H "Authorization: Bearer $TOKEN")

say() { echo "[install] $*"; }

# JSON without extra tools: JavaScript for Automation ships with macOS
jxa() { osascript -l JavaScript -e 'ObjC.import("Foundation"); var d = $.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile; var j = JSON.parse(ObjC.unwrap($.NSString.alloc.initWithDataEncoding(d, $.NSUTF8StringEncoding))); '"$1"; }

RELEASES="$(curl -fsSL "${auth[@]}" -H "Accept: application/vnd.github+json" -A mlt-install "https://api.github.com/repos/$REPO/releases?per_page=30")" \
    || { echo "Could not read the releases of $REPO. It is private: run 'gh auth login' once, or set GH_TOKEN to a token that can read it." >&2; exit 1; }
# one line per asset of the newest matching release:  tag<TAB>name<TAB>id<TAB>size
LIST="$(printf '%s' "$RELEASES" | jxa 'var want = "'"$TAG"'"; var r = j.filter(function(x){ return !x.draft && x.tag_name.indexOf("mac-intel-") === 0 && (want === "" || x.tag_name === want); })[0]; if (!r) { ""; } else { r.assets.map(function(a){ return r.tag_name + "\t" + a.name + "\t" + a.id + "\t" + a.size; }).join("\n"); }')"
[ -n "$LIST" ] || { echo "No Mac release (tag mac-intel-*) was found in $REPO." >&2; exit 1; }
say "release $(printf '%s\n' "$LIST" | head -1 | cut -f1)"

download() {   # name id size
    local name="$1" id="$2" size="$3" target="$WORK/$1"
    say "downloading $name ($((size / 1048576)) MB) ..."
    curl -fL --silent --show-error -C - --retry 40 --retry-all-errors --retry-delay 5 -A mlt-install "${auth[@]}" -H "Accept: application/octet-stream" \
        -o "$target" "https://api.github.com/repos/$REPO/releases/assets/$id"
    [ "$(stat -f %z "$target")" = "$size" ] || { echo "$name has the wrong size after the download." >&2; exit 1; }
}

PARTS=()
SHA_NAME=""
while IFS=$'\t' read -r _tag name id size; do
    case "$name" in
        *.dmg.part*) download "$name" "$id" "$size"; PARTS+=("$WORK/$name") ;;
        *Intel-Mac-Offline.dmg.sha256) SHA_NAME="$name"; download "$name" "$id" "$size" ;;
    esac
done <<EOF
$LIST
EOF
[ "${#PARTS[@]}" -gt 0 ] || { echo "The release has no DMG parts." >&2; exit 1; }
[ -n "$SHA_NAME" ] && [ -f "$WORK/$SHA_NAME" ] || { echo "The release has no .sha256 file." >&2; exit 1; }

DMG="$WORK/${SHA_NAME%.sha256}"
say "joining ${#PARTS[@]} parts ..."
IFS=$'\n' SORTED=($(printf '%s\n' "${PARTS[@]}" | sort)); unset IFS
cat "${SORTED[@]}" > "$DMG"
EXPECTED="$(awk '{print $1}' "$WORK/$SHA_NAME")"
ACTUAL="$(shasum -a 256 "$DMG" | awk '{print $1}')"
if [ "$EXPECTED" != "$ACTUAL" ]; then rm -f "$DMG"; echo "SHA-256 differs ($ACTUAL, expected $EXPECTED). The joined file was deleted; run this again." >&2; exit 1; fi
say "SHA-256 verified: $ACTUAL"
rm -f "${SORTED[@]}"

MOUNT="$(mktemp -d /tmp/mlt-dmg.XXXXXX)"
cleanup() { hdiutil detach "$MOUNT" -quiet >/dev/null 2>&1 || true; rmdir "$MOUNT" 2>/dev/null || true; }
trap cleanup EXIT
hdiutil attach "$DMG" -nobrowse -readonly -mountpoint "$MOUNT" -quiet
if [ ! -d "$MOUNT/$APP_NAME" ]; then                                                                      # prefer MELETRA.app, fall back to an earlier name
    for legacy in "${LEGACY_APP_NAMES[@]}"; do [ -d "$MOUNT/$legacy" ] && { APP_NAME="$legacy"; break; }; done
fi
[ -d "$MOUNT/$APP_NAME" ] || { echo "The disk image does not contain MELETRA.app." >&2; exit 1; }
APP_BASE="${APP_NAME%.app}"

mkdir -p "$DEST" 2>/dev/null || true
if [ ! -w "$DEST" ]; then DEST="$HOME/Applications"; mkdir -p "$DEST"; say "no write access to /Applications: installing into $DEST"; fi
OLD_COPIES=()                                            # copies under an earlier name: replaced by the new one (only when one really is this program)
for legacy in "${LEGACY_APP_NAMES[@]}"; do
    if [ "$legacy" != "$APP_NAME" ] && [ -d "$DEST/$legacy" ] && [ "$(defaults read "$DEST/$legacy/Contents/Info" CFBundleIdentifier 2>/dev/null || true)" = "$BUNDLE_ID" ]; then OLD_COPIES+=("$DEST/$legacy"); fi
done
for running in "$DEST/$APP_NAME" ${OLD_COPIES[@]+"${OLD_COPIES[@]}"}; do
    if pgrep -f "$running/Contents/MacOS" >/dev/null 2>&1; then
        say "closing the running copy ..."
        osascript -e "tell application \"$(basename "$running" .app)\" to quit" >/dev/null 2>&1 || true
        sleep 3
    fi
done
rm -rf "$DEST/$APP_NAME"
ditto "$MOUNT/$APP_NAME" "$DEST/$APP_NAME"
for old in ${OLD_COPIES[@]+"${OLD_COPIES[@]}"}; do rm -rf "$old"; say "replaced the earlier copy ($(basename "$old"))"; done
say "installed: $DEST/$APP_NAME"
hdiutil detach "$MOUNT" -quiet || true
trap - EXIT
rmdir "$MOUNT" 2>/dev/null || true
if [ "$KEEP" = "1" ]; then say "kept: $DMG"; else rm -f "$DMG" "$WORK/$SHA_NAME"; fi
say "done. Open it from Applications (Launchpad / Finder). The first start finds Whisper Large-v3 inside the app: no download, no internet needed."
