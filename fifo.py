from __future__ import annotations
from base import ReplacementAlgorithm, SupportsRequestSequence
from collections import deque
from collections.abc import Iterable
from typing import (
    Generator,
)


class FIFO(ReplacementAlgorithm, SupportsRequestSequence):
    """Implements the FIFO replacement algorithm.

    - Uses collections.deque as the queue, along with a set to keep track of the available pages.
    - Every operation works in O(1) constant time, except for request sequence which requires O(sequence length) time.   

    Args:
        max_pages (int): Takes the maximum number of pages.
    """

    def __init__(self, max_pages: int) -> None:
        assert max_pages > 0
        self._max_pages = max_pages
        self._pages: deque[int] = deque()
        self._available_pages: set[int] = set()
        self._count_page_faults = 0

    def request(self, page: int) -> None:
        """Request a page from the available pages.

        Args:
            page (int): The requested page.
        """

        if page in self._available_pages:
            return
        
        self._count_page_faults += 1

        if len(self._available_pages) < self._max_pages:
            self._available_pages.add(page)
            self._pages.append(page)
            return
        
        victim = self._pages.popleft()
        self._available_pages.remove(victim)
        self._available_pages.add(page)
        self._pages.append(page)

    def request_sequence(self, pages: Iterable[int]) -> None:
        """Request a sequence of pages from the available pages.

        Args:
            pages (Iterable[int]): sequence of pages.
        """

        for page in pages:
            self.request(page)

    def __iter__(self) -> Generator[int, None, None]:
        yield from (page for page in self._pages)
        
    @property
    def page_faults(self) -> int:
        """Number of page faults for the given requests

        Returns:
            int: Number of page faults.
        """

        return self._count_page_faults


def main() -> int:
    fifo = FIFO(max_pages=5)
    sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    fifo.request_sequence(sequence)

    print(f"{fifo.page_faults = }")
    print(*fifo, sep=' ', end='\n', flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
