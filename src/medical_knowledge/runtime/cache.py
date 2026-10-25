"""Small bounded caches for the run-time (frequent terms, recent concepts, recent fuzzy queries). Nothing here grows without a limit."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

V = TypeVar("V")


class LRU(Generic[V]):
    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = max(1, maxsize)
        self._data: "OrderedDict[Hashable, V]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, compute: Callable[[], V]) -> V:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        value = compute()
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {"size": len(self._data), "max": self.maxsize, "hits": self.hits, "misses": self.misses}
