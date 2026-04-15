from __future__ import annotations
from collections.abc import Iterable
from typing import (
    Protocol,
)


class ReplacementAlgorithm(Protocol):
    def request(self, frame: int) -> None: ...

    @property
    def page_faults(self) -> int: ...


class SupportsRequestSequence(Protocol):
    def request_sequence(self, frames: Iterable[int]) -> None: ...

    @property
    def page_faults(self) -> int: ...
