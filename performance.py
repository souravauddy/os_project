from __future__ import annotations
import matplotlib.pyplot as plt
import timeit
import random
from sympy import plot
import os
from algorithms.fifo import FIFO
from algorithms.lru import LRU
from algorithms.mfu import MFU
from algorithms.optimal import Optimal
from algorithms.second_chance import SecondChancePageReplacement
from algorithms.RL_state import ILCache
from algorithms.RL_nature import (
    load_model,
    LPRSimulator,
    LPRModel,
)
from typing import (
    Final,
    Generator,
    Iterable,
)


BASE_DIRECTORY: Final[str] = r"./algorithms/"
MAX_PAGES: Final[int] = int(64)


def generate_identities() -> Generator[int, None, None]:
    identity = 0

    while True:
        yield identity
        identity += 1


def generate_working_set_trace():
    trace = []

    for _ in range(100):
        working_set = random.sample(range(200), 30)

        for _ in range(300):
            if random.random() < 0.1:
                trace.append(random.randint(0, 200))
            else:
                trace.append(random.choice(working_set))

    return trace
    

def plot_page_faults(page_faults: dict[str, int], saving_directory: str = r"./plots/", filename: str = "page_faults_bar_plot") -> None:
    os.makedirs(saving_directory, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.bar(page_faults.keys(), page_faults.values())

    plt.xlabel("Algorithms")
    plt.ylabel("Page Faults")
    plt.title("Page fault across all the algorithms.")
    plt.legend()
    plt.grid(visible=True)

    saving_path = os.path.join(saving_directory, filename)
    plt.savefig(saving_path, dpi=300)
    plt.close()

    print(f"Plot saved at {saving_path}")


def plot_time_taken(times: dict[str, float], saving_directory: str = r"./plots/", filename: str = "time_taken") -> None:
    os.makedirs(saving_directory, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plt.bar(times.keys(), times.values())

    plt.xlabel("Algorithms")
    plt.ylabel("Time Taken")
    plt.title("Time taken across all the algorithms.")
    plt.legend()
    plt.grid(visible=True)

    saving_path = os.path.join(saving_directory, filename)
    plt.savefig(saving_path, dpi=300)
    plt.close()

    print(f"Plot saved at {saving_path}")


def _looping_trace(max_pages: int, n_refs: int = 15_000,
                   capacity: int = 10) -> list[int]:
    """Sequential scan patterns that are hard for LRU."""
    loop_size = capacity + random.randint(1, 5)   # just bigger than cache
    loop = random.sample(range(max_pages), min(loop_size, max_pages))
    trace: list[int] = []
    while len(trace) < n_refs:
        trace.extend(loop)
    return trace[:n_refs]


def main() -> int:
    os.chdir(BASE_DIRECTORY)
    reference_pages = []

    with open(r"./traces/trace_1.txt", "r+") as file:
        lines = [line.rstrip("\n") for line in file.readlines() if len(line.rstrip("\n")) > 0]
        reference_pages = list(map(int, lines))
        identity_generator = generate_identities()
        processed_pages = {}

        for item in reference_pages:
            if item in processed_pages:
                continue

            processed_pages[item] = next(identity_generator)

        for index, page in enumerate(reference_pages):
            reference_pages[index] = processed_pages[page]

    reference_pages = generate_working_set_trace()
    loop_reference_pages = _looping_trace(max_pages=200, capacity=64)
    print(f"unique pages = {len(set(reference_pages))}")
    print(len(reference_pages))

    fifo = FIFO(max_pages=MAX_PAGES)
    lru = LRU(max_pages=MAX_PAGES)
    optimal = Optimal(max_pages=MAX_PAGES)
    mfu = MFU(max_pages=MAX_PAGES)
    second_chance = SecondChancePageReplacement(capacity=MAX_PAGES)
    imitation = ILCache(max_pages=200, capacity=MAX_PAGES)

    fifo2 = FIFO(max_pages=MAX_PAGES)
    lru2 = LRU(max_pages=MAX_PAGES)
    optimal2 = Optimal(max_pages=MAX_PAGES)
    mfu2 = MFU(max_pages=MAX_PAGES)
    second_chance2 = SecondChancePageReplacement(capacity=MAX_PAGES)
    imitation2 = ILCache(max_pages=200, capacity=MAX_PAGES)

    model = load_model()
    model.history = {}
    model.time = 0
    simulator = LPRSimulator(model, MAX_PAGES)

    IL_time = timeit.timeit(lambda: imitation.request_sequence(reference_pages), number=1)
    fifo_time = timeit.timeit(lambda: fifo.request_sequence(reference_pages), number=1)
    lru_time = timeit.timeit(lambda: lru.request_sequence(reference_pages), number=1)
    optimal_time = timeit.timeit(lambda: optimal.request_sequence(reference_pages), number=1)
    mfu_time = timeit.timeit(lambda: mfu.request_sequence(reference_pages), number=1)
    RL_nature_page_faults = simulator.run(reference_pages)
    RL_nature_time = timeit.timeit(lambda: simulator.run(reference_pages), number=1)
    second_chance_time = timeit.timeit(lambda: second_chance.request_sequence(reference_pages), number=1)
    IL_page_faults = imitation.request_sequence(reference_pages)

    fifo_time2 = timeit.timeit(lambda: fifo2.request_sequence(loop_reference_pages), number=1)
    lru_time2 = timeit.timeit(lambda: lru2.request_sequence(loop_reference_pages), number=1)
    optimal_time2 = timeit.timeit(lambda: optimal2.request_sequence(loop_reference_pages), number=1)
    mfu_time2 = timeit.timeit(lambda: mfu2.request_sequence(loop_reference_pages), number=1)
    RL_nature_page_faults2 = simulator.run(loop_reference_pages)
    RL_nature_time2 = timeit.timeit(lambda: simulator.run(loop_reference_pages), number=1)
    second_chance_time2 = timeit.timeit(lambda: second_chance2.request_sequence(loop_reference_pages), number=1)
    IL_page_faults2 = imitation2.request_sequence(loop_reference_pages)
    IL_time2 = timeit.timeit(lambda: imitation2.request_sequence(loop_reference_pages), number=1)
    
    times: dict[str, float] = {
        "fifo": fifo_time * 1000,
        "lru": lru_time * 1000,
        "optimal": optimal_time * 1000,
        "mfu": mfu_time * 1000,
        "RL_nature": RL_nature_time * 1000,
        "second_chance": second_chance_time * 1000,
        "IL": IL_time * 1000,
    }

    times2: dict[str, float] = {
        "fifo": fifo_time2 * 1000,
        "lru": lru_time2 * 1000,
        "optimal": optimal_time2 * 1000,
        "mfu": mfu_time2 * 1000,
        "RL_nature": RL_nature_time2 * 1000,
        "second_chance": second_chance_time2 * 1000,
        "IL": IL_time2 * 1000,
    }

    page_faults2: dict[str, int] = {
        "fifo": fifo2.page_faults,
        "lru": lru2.page_faults,
        "optimal": optimal2.page_faults,
        "mfu": mfu2.page_faults,
        "RL_nature": RL_nature_page_faults2,
        "second_chance": second_chance2.page_faults,
        "IL": IL_page_faults2,
    }

    page_faults: dict[str, int] = {
        "fifo": fifo.page_faults,
        "lru": lru.page_faults,
        "optimal": optimal.page_faults,
        "mfu": mfu.page_faults,
        "RL_nature": RL_nature_page_faults,
        "second_chance": second_chance.page_faults,
        "IL": IL_page_faults,
    }

    os.chdir("..")

    print(page_faults)
    plot_page_faults(page_faults)
    plot_page_faults(page_faults2, filename="page faults on looping data.")
    plot_time_taken(times)
    plot_time_taken(times2, filename="time taken on looping data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
