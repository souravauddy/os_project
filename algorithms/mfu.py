from __future__ import annotations
import collections
from collections.abc import Iterable
from .base import ReplacementAlgorithm, SupportsRequestSequence
from typing import (
    Generator,
)


class MFU(ReplacementAlgorithm, SupportsRequestSequence):
    """Implementation of the Most Frequenly Used page replacement algorithm.
    - Always create a new instance if you want to reset.
    - All of the page faults will be accumulated.

    Args:
        max_pages (int): maximum number of pages that has been allocated.
    """

    def __init__(self, max_pages: int) -> None:
        self._max_pages = max_pages
        self._count_page_faults = 0
        self._counter: collections.defaultdict[int, int] = collections.defaultdict(lambda : 0)
    
    def request(self, page: int) -> None:
        """Request a page form the available pages.

        Args:
            page (int): Requested page.
        """

        if page in self._counter:
            self._counter[page] += 1
            return
        
        self._count_page_faults += 1
        
        if len(self._counter) < self._max_pages:
            self._counter[page] += 1
            return
        
        victim = max(self._counter.keys(), key=lambda x : self._counter[x])
        self._counter.pop(victim)
        self._counter[page] += 1

    def request_sequence(self, pages: Iterable[int]) -> None:
        """Request a sequence of pages.

        Args:
            pages (Iterable[int]): An Iterable instance of the requested pages.
        """

        for page in pages:
            self.request(page)

    def __iter__(self) -> Generator[int, None, None]:
        yield from self._counter.keys()

    @property
    def page_faults(self) -> int:
        """The number of page faults.

        Returns:
            int: number of page faults.
        """

        return self._count_page_faults


def main() -> int:
    mfu = MFU(5)
    pages = [1, 2, 4, 2, 4, 4, 5, 2, 9, 8, 8, 9, 2, 1, 3, 4, 5, 7, 8, 2, 20, 15]
    mfu.request_sequence(pages)

    print(mfu.page_faults)
    print(*mfu)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
