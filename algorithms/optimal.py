from __future__ import annotations
import collections
from collections import deque
from .base import SupportsRequestSequence
from collections.abc import Iterable
from typing import (
    Final,
    Generator,
)


class Optimal(SupportsRequestSequence):
    """Implementation of the Optimal Algorithm for the page replacement.
    - When, a new request sequence has to be given the reinitialize method has to be called first or create a new instance should be created.
    - The reinitialize method is only a method available in the optimal algorithm because the algorithm works only for a sequence.
    - The remainder algorithms both work on sequence as well as single page requests.
    - Time complexity is: O(max frames * sequence length), but could be reduced to O(log(max frames) * sequence length) using a binary heap.

    Args:
        max_pages (int): Takes the number of pages allocated to the process.
    """

    __slots__ = ("_max_pages", "_count_page_faults", "_available_pages")
    _INF: Final[int] = int(1e9 + 5)
    _FRONT: Final[int] = int(0)

    @staticmethod
    def _process(locations: dict[int, deque[int]], victim: int) -> None:
        if len(locations[victim]) == 0:
            locations.pop(victim)

    def __init__(self, max_pages: int) -> None:
        assert max_pages > 0
        self._max_pages = max_pages
        self._count_page_faults: int = 0
        self._available_pages: set[int] = set()

    def reinitialize(self) -> None:
        """
        Reinitialzes the class so that a new request sequence can be used.
        """

        self._count_page_faults = 0
        self._available_pages.clear()

    def request_sequence(self, pages: Iterable[int]) -> None:
        """Requests a sequence of pages, and calcuates the number of page faults.
        Complexity is O(number of pages * max_pages).
        Since, number of pages >> max_pages this works better then the O(number_of_pages * number_of_pages) naive complexity.

        Args:
            pages (Iterable[int]): Requested pages.
        """

        locations: dict[int, deque[int]] = collections.defaultdict(deque)

        for index, page in enumerate(pages):
            locations[page].append(index)

        for key in locations.keys():
            locations[key].append(self._INF)

        for requested_page in pages:
            locations[requested_page].popleft()

            if requested_page in self._available_pages:
                continue

            self._count_page_faults += 1

            if len(self._available_pages) < self._max_pages:
                self._available_pages.add(requested_page)
                continue
            
            victim = max(self._available_pages, key=lambda x: locations[x][self._FRONT])
            self._process(locations, victim)
            self._available_pages.remove(victim)
            self._available_pages.add(requested_page)

    def __iter__(self) -> Generator[int, None, None]:
        yield from (page for page in self._available_pages)

    @property
    def page_faults(self) -> int:
        """Returns the number of page faults occurred in the given sequence.

        Returns:
            int: number of page faults.
        """

        return self._count_page_faults


def main() -> int:
    optimal = Optimal(5)
    sequence = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 6, 7, 8, 1, 2, 3, 4, 5]
    optimal.request_sequence(sequence)
    print(optimal.page_faults)

    for page in optimal:
        print(page)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
