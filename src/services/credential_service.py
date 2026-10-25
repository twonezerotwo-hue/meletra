"""API keys in the operating system's secure store (never in a config file, never in a log).

* Windows: ``WindowsCredentialBackend`` talks to ``advapi32`` (CredWriteW / CredReadW / CredDeleteW) through ctypes, so no extra
  package is needed; Windows keeps the secret under the current user's account.
* macOS: ``KeyringBackend`` (the ``keyring`` package, when it is installed: Keychain API, nothing on a command line) or
  ``MacKeychainBackend`` (the ``security`` command line tool).
* Anything else and the tests: ``MemoryCredentialBackend``.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import Any, Callable, Optional, Protocol

try:
    from ctypes import wintypes
except (ImportError, ValueError):                 # pragma: no cover - some macOS / Linux builds
    class wintypes:                               # type: ignore[no-redef]
        """Stand-ins so the Windows structures below can be declared everywhere; they are only ever used on Windows."""

        DWORD, BOOL = ctypes.c_ulong, ctypes.c_int
        LPWSTR = LPCWSTR = ctypes.c_wchar_p

from .. import APP_ID

DEEPGRAM_TARGET = f"{APP_ID}:deepgram-api-key"
CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168


class CredentialBackend(Protocol):
    def set(self, target: str, secret: str) -> None: ...
    def get(self, target: str) -> Optional[str]: ...
    def delete(self, target: str) -> bool: ...


class MemoryCredentialBackend:
    """For tests and non-Windows systems: lives only as long as the process."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set(self, target: str, secret: str) -> None:
        self._store[target] = secret

    def get(self, target: str) -> Optional[str]:
        return self._store.get(target)

    def delete(self, target: str) -> bool:
        return self._store.pop(target, None) is not None


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR), ("LastWritten", _FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


class WindowsCredentialBackend:
    label = "Windows Credential Manager"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows Credential Manager is only available on Windows")
        self._advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        self._advapi.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
        self._advapi.CredWriteW.restype = wintypes.BOOL
        self._advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                           ctypes.POINTER(ctypes.POINTER(_CREDENTIALW))]
        self._advapi.CredReadW.restype = wintypes.BOOL
        self._advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self._advapi.CredDeleteW.restype = wintypes.BOOL
        self._advapi.CredFree.argtypes = [ctypes.c_void_p]

    def set(self, target: str, secret: str) -> None:
        blob = secret.encode("utf-16-le")
        buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        credential = _CREDENTIALW()
        credential.Type = CRED_TYPE_GENERIC
        credential.TargetName = target
        credential.CredentialBlobSize = len(blob)
        credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = os.environ.get("USERNAME", "user")
        if not self._advapi.CredWriteW(ctypes.byref(credential), 0):
            raise OSError(f"Windows Credential Manager refused to store the key (error {ctypes.get_last_error()})")

    def get(self, target: str) -> Optional[str]:
        pointer = ctypes.POINTER(_CREDENTIALW)()
        if not self._advapi.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            return None
        try:
            credential = pointer.contents
            raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return raw.decode("utf-16-le")
        finally:
            self._advapi.CredFree(pointer)

    def delete(self, target: str) -> bool:
        return bool(self._advapi.CredDeleteW(target, CRED_TYPE_GENERIC, 0))


class MacKeychainBackend:
    """The login keychain through the ``security`` tool (generic password: account = target, service = the app)."""

    label = "macOS Keychain"
    SERVICE = APP_ID

    def __init__(self, run: Optional[Callable[..., Any]] = None) -> None:
        self._run = run or subprocess.run

    def _security(self, *args: str) -> Any:
        return self._run(["security", *args], capture_output=True, text=True, check=False, timeout=15)

    def set(self, target: str, secret: str) -> None:
        # -U updates an existing item. (The secret is briefly visible to other processes of the same user as a command argument;
        # KeyringBackend avoids that and is preferred whenever the keyring package is installed.)
        result = self._security("add-generic-password", "-a", target, "-s", self.SERVICE, "-U", "-w", secret)
        if result.returncode != 0:
            raise OSError("The macOS Keychain refused to store the key (" + (result.stderr or "").strip()[:120] + ")")

    def get(self, target: str) -> Optional[str]:
        result = self._security("find-generic-password", "-a", target, "-s", self.SERVICE, "-w")
        if result.returncode != 0:
            return None
        return (result.stdout or "").rstrip(chr(10)) or None

    def delete(self, target: str) -> bool:
        return self._security("delete-generic-password", "-a", target, "-s", self.SERVICE).returncode == 0


class KeyringBackend:
    """The ``keyring`` package (macOS Keychain API). Created only when the package can be imported."""

    label = "macOS Keychain"

    def __init__(self, module: Any) -> None:
        self._keyring = module

    @classmethod
    def try_create(cls) -> Optional["KeyringBackend"]:
        try:
            import keyring                         # type: ignore[import-not-found]
        except Exception:                          # noqa: BLE001 - not installed / no backend: use the command line tool
            return None
        return cls(keyring)

    def set(self, target: str, secret: str) -> None:
        self._keyring.set_password(APP_ID, target, secret)

    def get(self, target: str) -> Optional[str]:
        return self._keyring.get_password(APP_ID, target) or None

    def delete(self, target: str) -> bool:
        try:
            self._keyring.delete_password(APP_ID, target)
        except Exception:                          # noqa: BLE001 - nothing stored
            return False
        return True


def default_backend() -> CredentialBackend:
    from ..platform import current

    return current().credential_backend()


class CredentialService:
    def __init__(self, backend: Optional[CredentialBackend] = None, target: str = DEEPGRAM_TARGET) -> None:
        self.backend = backend or default_backend()
        self.target = target

    @property
    def secure_storage(self) -> bool:
        """True when the key really lives in the operating system's secure store (not just in memory)."""
        return not isinstance(self.backend, MemoryCredentialBackend)

    @property
    def store_label(self) -> str:
        """Where the key is kept, for the settings text ("Windows Credential Manager", "macOS Keychain", "memory only")."""
        return getattr(self.backend, "label", "memory only (this system has no secure store)")

    def set_deepgram_key(self, key: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("The API key is empty.")
        if any(ch.isspace() for ch in key):
            raise ValueError("An API key does not contain spaces or line breaks.")
        self.backend.set(self.target, key)

    def get_deepgram_key(self) -> Optional[str]:
        return self.backend.get(self.target) or None

    def delete_deepgram_key(self) -> bool:
        return self.backend.delete(self.target)

    def has_deepgram_key(self) -> bool:
        return self.get_deepgram_key() is not None

    # ---- keys of the AI Study providers (cloud / OpenAI-compatible); the built-in provider and Ollama need none
    @staticmethod
    def study_target(provider: str) -> str:
        return f"{APP_ID}:study-{provider}-api-key"

    def set_study_key(self, provider: str, key: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("The API key is empty.")
        if any(ch.isspace() for ch in key):
            raise ValueError("An API key does not contain spaces or line breaks.")
        self.backend.set(self.study_target(provider), key)

    def get_study_key(self, provider: str) -> Optional[str]:
        return self.backend.get(self.study_target(provider)) or None

    def delete_study_key(self, provider: str) -> bool:
        return self.backend.delete(self.study_target(provider))

    def masked_study_key(self, provider: str) -> str:
        key = self.get_study_key(provider)
        if not key:
            return ""
        return "•" * 8 + key[-4:] if len(key) > 8 else "•" * len(key)

    def masked_deepgram_key(self) -> str:
        key = self.get_deepgram_key()
        if not key:
            return ""
        return "•" * 8 + key[-4:] if len(key) > 8 else "•" * len(key)
