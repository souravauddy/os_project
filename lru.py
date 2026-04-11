from __future__ import annotations
from base import SupportsRequestSequence, ReplacementAlgorithm
from collections.abc import Iterable
from typing import (
    Generator,
    Final,
)


class Node[T]:
    __slots__ = ("value", "left", "right")

    def __init__(self, value: T, left: Node[T] | None = None, right: Node[T] | None = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Node(value={self.value!r})"


import dataclasses

@dataclasses.dataclass(slots=True, repr=True, init=True)
class Node[T]:
    value: T
    left: Node[T] | None = None
    right: Node[T] | None = None


class LRU(ReplacementAlgorithm, SupportsRequestSequence):
    """Implementation of the LRU page replacement algorithm.
    Uses the doubly linked list and hashmap implementation of the LRU cache.
    It takes in the number of pages as input.
    All the methods work in O(1) time.
    This class cannot be reinitialzed unlike the optimal algorithm immplementation always create a new instance.

    Args:
        max_pages (int): The maximum number of pages in the cache.
    """

    _SENTINEL: Final[int] = int(2e9 + 5)

    def __init__(self, max_pages: int) -> None:
        assert max_pages > 0
        self._available_pages: dict[int, Node[int]] = {}
        self._head: Node[int] = Node(self._SENTINEL)
        self._tail: Node[int] = Node(self._SENTINEL)
        self._head.right = self._tail
        self._tail.left = self._head
        self._max_pages = max_pages
        self._count_page_faults = 0

    @property
    def page_faults(self) -> int:
        """Returns the number of page faults occurred.

        Returns:
            int: The number of page faults.
        """

        return self._count_page_faults
    
    def _detach(self, node: Node[int]) -> Node[int]:
        assert node.left is not None and node.right is not None
        node.left.right = node.right
        node.right.left = node.left
        node.left = node.right = None
        return node
    
    def _attach(self, node: Node[int]) -> None:
        assert self._tail.left is not None
        self._tail.left.right = node
        node.left = self._tail.left
        self._tail.left = node
        node.right = self._tail

    def empty(self) -> bool:
        """Checks if some pages have not been allocated does not clear the pages.

        Returns:
            bool: return True if none of the pages have been allocated else False.
        """

        return len(self._available_pages) == 0
    
    def request(self, page_no: int) -> None:
        """Receives the page no the user has requested for, and makes the necessary adjustments.

        Args:
            page_no (int): requested page.
        """

        if page_no in self._available_pages:
            node = self._detach(self._available_pages[page_no])
            self._attach(node)
            return
        
        new_node = Node(page_no)
        self._count_page_faults += 1

        if len(self._available_pages) < self._max_pages:
            self._attach(new_node)
            self._available_pages[page_no] = new_node
            return
        
        assert self._head.right is not None

        victim = self._detach(self._head.right)
        self._available_pages.pop(victim.value)
        self._attach(new_node)
        self._available_pages[page_no] = new_node

    def request_sequence(self, pages: Iterable[int]) -> None:
        """Receives a list of pages as a request.

        Args:
            pages (Iterable[int]): pages that are requested.
        """

        for page in pages:
            self.request(page)

    def __iter__(self) -> Generator[int, None, None]:
        iterator = self._head.right

        while iterator is not self._tail:
            assert iterator is not None
            yield iterator.value
            iterator = iterator.right


def main() -> int:
    max_pages = int(input("Enter the number of pages: "))
    pages = map(int, input().split())
    lru = LRU(max_pages)
    lru.request_sequence(pages)

    print(f"{lru.page_faults = }")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
