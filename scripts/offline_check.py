"""Does the PACKAGED program ever talk to the network while it works? The full packaged smoke test, watched.

    python scripts/offline_check.py --exe dist/v4.1.0/MELETRA/MELETRA.exe --model-dir C:/.../models --out OUT_DIR [--quick]

The packaged program runs through ``scripts/smoke_packaged.py`` (self-test, window, diagnostics, English speech recognition on the GPU and on the CPU, video and audio files of
several formats, a document queue, Study, pause / resume) with a PATH that holds only Windows, ``HF_HUB_OFFLINE=1`` and proxy variables that point nowhere, so an attempt to reach the
internet would fail instead of succeed. Meanwhile the computer's TCP table (what ``netstat`` prints, read through the Windows API) is read every 0.05 s and every connection of every process that runs from the program's own folder is recorded; the
check passes when no connection to an address outside this computer was ever seen. (A sampler can miss a connection that lives for less than about 0.1 s; the strict proof is the
firewall-blocked run of the release workflow on the public bytes. This computer's session has no administrator right to block the network itself.)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ("127.0.0.1", "::1", "[::1]", "0.0.0.0", "[::]", "[::]", "localhost")


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TCP_TABLE_OWNER_PID_ALL, AF_INET, AF_INET6 = 5, 2, 23
STATES = {1: "CLOSED", 2: "LISTENING", 3: "SYN_SENT", 4: "SYN_RCVD", 5: "ESTABLISHED", 6: "FIN_WAIT1", 7: "FIN_WAIT2", 8: "CLOSE_WAIT", 9: "CLOSING", 10: "LAST_ACK", 11: "TIME_WAIT", 12: "DELETE_TCB"}


def program_pids(folder: Path) -> set[int]:
    """The ids of the processes that run a file from ``folder`` (the program and its workers), read with the Windows API in a millisecond."""
    ids = (wintypes.DWORD * 4096)()
    used = wintypes.DWORD()
    if not psapi.EnumProcesses(ids, ctypes.sizeof(ids), ctypes.byref(used)):
        return set()
    prefix = str(folder).lower().rstrip("\\") + "\\"
    found = set()
    for pid in ids[: used.value // ctypes.sizeof(wintypes.DWORD)]:
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            continue
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) and buffer.value.lower().startswith(prefix):
                found.add(pid)
        finally:
            kernel32.CloseHandle(handle)
    return found


def _table(family: int) -> bytes:
    size = wintypes.DWORD(0)
    iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False, family, TCP_TABLE_OWNER_PID_ALL, 0)
    for _ in range(5):
        buffer = ctypes.create_string_buffer(size.value + 4096)
        size = wintypes.DWORD(len(buffer))
        if iphlpapi.GetExtendedTcpTable(buffer, ctypes.byref(size), False, family, TCP_TABLE_OWNER_PID_ALL, 0) == 0:
            return buffer.raw
    return b""


def netstat_rows() -> list[tuple[str, str, str, str, int]]:
    """Every TCP connection of the computer (IPv4 and IPv6): (protocol, local, remote, state, owning process id) - the table that netstat prints, read directly."""
    rows = []
    raw = _table(AF_INET)
    if raw:
        count = struct.unpack_from("<I", raw, 0)[0]
        for i in range(count):
            state, laddr, lport, raddr, rport, pid = struct.unpack_from("<6I", raw, 4 + 24 * i)
            rows.append(("tcp", f"{socket.inet_ntoa(struct.pack('<I', laddr))}:{socket.ntohs(lport & 0xFFFF)}", f"{socket.inet_ntoa(struct.pack('<I', raddr))}:{socket.ntohs(rport & 0xFFFF)}", STATES.get(state, str(state)), pid))
    raw = _table(AF_INET6)
    if raw:
        count = struct.unpack_from("<I", raw, 0)[0]
        for i in range(count):
            offset = 4 + 56 * i
            laddr = raw[offset:offset + 16]
            lport = struct.unpack_from("<I", raw, offset + 20)[0]
            raddr = raw[offset + 24:offset + 40]
            rport = struct.unpack_from("<I", raw, offset + 44)[0]
            state, pid = struct.unpack_from("<2I", raw, offset + 48)
            rows.append(("tcp", f"[{socket.inet_ntop(socket.AF_INET6, laddr)}]:{socket.ntohs(lport & 0xFFFF)}", f"[{socket.inet_ntop(socket.AF_INET6, raddr)}]:{socket.ntohs(rport & 0xFFFF)}", STATES.get(state, str(state)), pid))
    return rows


def remote_host(address: str) -> str:
    return address.rsplit(":", 1)[0]


class Watcher:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.pids: set[int] = set()
        self.seen_pids: set[int] = set()
        self.samples = 0
        self.outside: dict[str, dict] = {}
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self._pids, daemon=True), threading.Thread(target=self._connections, daemon=True)]

    def _pids(self) -> None:
        while not self.stop.is_set():
            try:
                self.pids = program_pids(self.folder)
                self.seen_pids |= self.pids
            except Exception:                                                                  # noqa: BLE001 - a failed sample is not a result
                pass
            self.stop.wait(0.3)

    def _connections(self) -> None:
        while not self.stop.is_set():
            try:
                for proto, local, foreign, state, pid in netstat_rows():
                    if pid in self.pids and state != "LISTENING" and remote_host(foreign) not in LOCAL:
                        self.outside[f"{foreign} (pid {pid})"] = {"proto": proto, "local": local, "state": state, "at": time.strftime("%H:%M:%S")}
                self.samples += 1
            except Exception:                                                                  # noqa: BLE001
                pass
            self.stop.wait(0.05)

    def __enter__(self) -> "Watcher":
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="start-up checks only (no model needed)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)                      # HIGH_PRIORITY_CLASS: the watcher must keep reading while the program uses every core
    folder = args.exe.resolve().parent
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9", NO_PROXY="")
    report_file = args.out / "offline_smoke_report.json"
    command = [sys.executable, str(ROOT / "scripts" / "smoke_packaged.py"), "--exe", str(args.exe), "--model-dir", str(args.model_dir), "--report", str(report_file)]
    command.append("--skip-download")                                                       # the one step that is SUPPOSED to use the internet
    if args.quick:
        command.append("--quick")
    started = time.monotonic()
    with Watcher(folder) as watch:
        done = subprocess.run(command, env=env, cwd=str(ROOT), capture_output=True, text=True, errors="replace")
    smoke = json.loads(report_file.read_text(encoding="utf-8")) if report_file.is_file() else {}
    steps = smoke.get("steps", [])
    checks = {
        "the packaged smoke test passed (exit code 0)": done.returncode == 0,
        "no connection to an address outside this computer was seen": not watch.outside,
        "the watcher really saw the program, for long enough (4+ processes, 200+ samples, 3+ minutes: a real transcription ran)": len(watch.seen_pids) >= 4 and watch.samples >= 200 and time.monotonic() - started >= 180,
    }
    result = {"exe": str(args.exe), "seconds": round(time.monotonic() - started), "samples": watch.samples, "processes_seen": len(watch.seen_pids), "outside_connections": watch.outside,
              "steps": [{"name": s.get("name"), "ok": s.get("ok"), "detail": str(s.get("detail", ""))[:160]} for s in steps], "checks": checks, "ok": all(checks.values()),
              "smoke_tail": done.stdout[-1500:]}
    (args.out / "offline_check_report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    print(f"{watch.samples} samples, {len(watch.seen_pids)} processes, outside connections: {sorted(watch.outside) or 'none'}")
    print("OFFLINE_CHECK_OK" if result["ok"] else "OFFLINE_CHECK_FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
