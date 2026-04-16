"""
il_cache.py  —  Imitation Learning Page Replacement (mimics OPT)
=================================================================

How it works
------------
1. COLLECT: Run Bélády's OPT on many synthetic traces. At every eviction
   decision record the cache state + which slot OPT chose to evict.
   This gives us labelled (state, label) pairs with no contradictory signal.

2. TRAIN: Supervised multi-class classification.
   Input  : feature matrix of shape (capacity, FEATURE_DIM) — one row per slot.
   Output : probability distribution over slots → argmax = evict that slot.
   Loss   : cross-entropy against OPT's choice.

3. INFER: At runtime, build the same feature matrix (without future knowledge)
   and pick the slot the model scores highest for eviction.

Features per cached slot (all normalised to [0,1])
---------------------------------------------------
  0  page_id          / max_pages
  1  recency          (time since last use / horizon)
  2  frequency        (use count / horizon)
  3  inter_arrival    (average gap between uses, estimated from history / horizon)

The key insight: even without knowing the future, the model can learn that
pages with HIGH recency and LOW frequency tend to have far-away next uses —
which is exactly the OPT heuristic expressed in observable features.

Usage
-----
    from il_cache import ILCache

    cache = ILCache(max_pages=200, capacity=10)
    cache.train()                        # collects data from OPT + trains
    faults = cache.request_sequence(my_sequence)
    print(f"Page faults: {faults}")
"""

import os
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "il_cache_model.pt"
FEATURE_DIM = 4
HORIZON     = 500.0   # normalisation constant for time-based features


# ─────────────────────────────────────────────────────────────────────────────
# Network  —  slot scorer turned into a classifier
# ─────────────────────────────────────────────────────────────────────────────
class EvictionNet(nn.Module):
    """
    Input : (capacity, FEATURE_DIM)  — one feature row per cached slot
    Output: (capacity,)              — logit for "evict this slot"
    Training loss: cross-entropy with OPT's chosen slot as the label.
    """
    
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        # Shared per-slot encoder
        self.slot_enc = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Global context: pool over all slots then broadcast back
        self.context = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Final scorer per slot (uses local + global features)
        self.scorer = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, capacity, FEATURE_DIM)  →  logits: (B, capacity)"""
        B, C, _ = x.shape
        slot_feats = self.slot_enc(x)                    # (B, C, 128)
        global_ctx = slot_feats.mean(dim=1)              # (B, 128)
        global_ctx = self.context(global_ctx)            # (B, 64)
        global_ctx = global_ctx.unsqueeze(1).expand(B, C, -1)  # (B, C, 64)
        combined   = torch.cat([slot_feats, global_ctx], dim=-1)  # (B, C, 192)
        logits     = self.scorer(combined).squeeze(-1)   # (B, C)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Trace generators
# ─────────────────────────────────────────────────────────────────────────────
def _working_set_trace(max_pages: int, n_refs: int = 15_000) -> list[int]:
    trace: list[int] = []
    ws_size = max(5, max_pages // 7)
    for _ in range(max(1, n_refs // 300)):
        ws = random.sample(range(max_pages), min(ws_size, max_pages))
        for _ in range(300):
            trace.append(
                random.randint(0, max_pages - 1)
                if random.random() < 0.08
                else random.choice(ws)
            )
    return trace


def _zipf_trace(max_pages: int, n_refs: int = 15_000) -> list[int]:
    """Zipf-distributed accesses — hot pages get hit much more often."""
    ranks = np.arange(1, max_pages + 1, dtype=np.float64)
    probs = (1.0 / ranks) / (1.0 / ranks).sum()
    pages = np.random.choice(max_pages, size=n_refs, p=probs).tolist()
    return pages


def _looping_trace(max_pages: int, n_refs: int = 15_000,
                   capacity: int = 10) -> list[int]:
    """Sequential scan patterns that are hard for LRU."""
    loop_size = capacity + random.randint(1, 5)   # just bigger than cache
    loop = random.sample(range(max_pages), min(loop_size, max_pages))
    trace: list[int] = []
    while len(trace) < n_refs:
        trace.extend(loop)
    return trace[:n_refs]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
class FeatureTracker:
    """Maintains per-page statistics as we walk through a trace."""

    def __init__(self, max_pages: int):
        self.max_pages   = max_pages
        self.time        = 0
        self.last_used   = {}
        self.freq        = defaultdict(int)
        self.prev_access = {}   # for inter-arrival estimation
        self.inter_arr   = {}   # running mean inter-arrival

    def update(self, page: int):
        self.time += 1
        if page in self.prev_access:
            gap = self.time - self.prev_access[page]
            old = self.inter_arr.get(page, gap)
            self.inter_arr[page] = 0.7 * old + 0.3 * gap   # EMA
        self.prev_access[page] = self.time
        self.last_used[page]   = self.time
        self.freq[page]       += 1

    def features(self, page: int) -> list[float]:
        rec  = (self.time - self.last_used.get(page, 0)) / HORIZON
        freq = min(self.freq[page], HORIZON) / HORIZON
        ia   = min(self.inter_arr.get(page, HORIZON), HORIZON) / HORIZON
        return [page / self.max_pages, rec, freq, ia]

    def cache_state(self, cache: list[int]) -> np.ndarray:
        rows = [self.features(p) for p in cache]
        return np.array(rows, dtype=np.float32)   # (len(cache), FEATURE_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# OPT oracle
# ─────────────────────────────────────────────────────────────────────────────
def _opt_evict_idx(cache: list[int], trace: list[int], pos: int) -> int:
    """Return index in `cache` of the page with the farthest next use."""
    def next_use(page):
        for i in range(pos + 1, len(trace)):
            if trace[i] == page:
                return i
        return float('inf')
    return max(range(len(cache)), key=lambda i: next_use(cache[i]))


# ─────────────────────────────────────────────────────────────────────────────
# Data collection: run OPT and record (state, label) pairs
# ─────────────────────────────────────────────────────────────────────────────
def collect_demonstrations(max_pages: int, capacity: int,
                            n_traces: int = 60,
                            refs_per_trace: int = 15_000
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, capacity, FEATURE_DIM)  — cache state at each eviction
    y : (N,)                        — slot index OPT chose to evict
    """
    X_list, y_list = [], []

    trace_fns = [
        lambda: _working_set_trace(max_pages, refs_per_trace),
        lambda: _zipf_trace(max_pages, refs_per_trace),
        lambda: _looping_trace(max_pages, refs_per_trace, capacity),
    ]

    for t in range(n_traces):
        fn    = trace_fns[t % len(trace_fns)]
        trace = fn()
        cache: list[int] = []
        tracker = FeatureTracker(max_pages)

        for pos, page in enumerate(trace):
            tracker.update(page)

            if page in cache:
                continue   # hit — no decision needed

            if len(cache) < capacity:
                cache.append(page)
                continue   # cold miss, no eviction yet

            # Full cache — OPT decides
            evict_idx = _opt_evict_idx(cache, trace, pos)
            state     = tracker.cache_state(cache)  # (capacity, FEATURE_DIM)

            X_list.append(state)
            y_list.append(evict_idx)

            cache[evict_idx] = page

        if (t + 1) % 10 == 0:
            print(f"  Collected {t+1}/{n_traces} traces "
                  f"({len(X_list):,} eviction samples so far)")

    X = np.stack(X_list).astype(np.float32)   # (N, capacity, FEATURE_DIM)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Baseline algorithms (for benchmarking)
# ─────────────────────────────────────────────────────────────────────────────
def lru_faults(reference: list[int], capacity: int) -> int:
    cache: list[int] = []
    last: dict[int, int] = {}
    faults = 0
    for t, p in enumerate(reference):
        if p in cache:
            cache.remove(p); cache.append(p)
        else:
            faults += 1
            if len(cache) >= capacity:
                cache.remove(min(cache, key=lambda x: last.get(x, 0)))
            cache.append(p)
        last[p] = t
    return faults


def opt_faults(reference: list[int], capacity: int) -> int:
    cache: set[int] = set()
    cache_list: list[int] = []
    faults = 0
    for t, p in enumerate(reference):
        if p in cache:
            continue
        faults += 1
        if len(cache) < capacity:
            cache.add(p); cache_list.append(p)
        else:
            idx    = _opt_evict_idx(cache_list, reference, t)
            evicted = cache_list[idx]
            cache.remove(evicted)
            cache_list[idx] = p
            cache.add(p)
    return faults


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class ILCache:
    """
    Imitation-Learning page replacement.

    Parameters
    ----------
    max_pages  : total number of distinct page IDs in the system
    capacity   : number of cache frames
    model_path : where to persist weights
    """

    def __init__(self, max_pages: int = 200, capacity: int = 10,
                 model_path: str = MODEL_PATH):
        self.max_pages  = max_pages
        self.capacity   = capacity
        self.model_path = model_path
        self.model      = EvictionNet(capacity)
        self._trained   = self._try_load()

    # ── persistence ───────────────────────────────────────────────────
    def _try_load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        ckpt = torch.load(self.model_path, weights_only=True)
        if (ckpt.get("max_pages") == self.max_pages and
                ckpt.get("capacity") == self.capacity):
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            print(f"[ILCache] Loaded weights from '{self.model_path}'")
            return True
        print("[ILCache] Saved weights are for different params — retraining.")
        return False

    def _save(self):
        torch.save({
            "model":     self.model.state_dict(),
            "max_pages": self.max_pages,
            "capacity":  self.capacity,
        }, self.model_path)
        print(f"[ILCache] Weights saved to '{self.model_path}'")

    # ── training ──────────────────────────────────────────────────────
    def train(self, n_traces: int = 60, epochs: int = 30,
              batch_size: int = 512, lr: float = 1e-3,
              force: bool = False):
        """
        Collect OPT demonstrations then train via cross-entropy.

        Parameters
        ----------
        n_traces   : number of synthetic traces to run OPT on
        epochs     : supervised training epochs
        batch_size : mini-batch size
        force      : retrain even if weights exist
        """
        if self._trained and not force:
            print("[ILCache] Skipping training — loaded existing weights.")
            print("          Pass force=True to retrain.")
            return

        # ── Step 1: collect demonstrations ────────────────────────────
        print("\n[1/2] Collecting OPT demonstrations …")
        X, y = collect_demonstrations(self.max_pages, self.capacity,
                                      n_traces=n_traces)
        print(f"      Dataset: {len(X):,} eviction decisions collected\n")

        # ── Step 2: supervised training ───────────────────────────────
        print("[2/2] Training EvictionNet …")
        dataset    = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader     = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, drop_last=False)
        opt        = optim.AdamW(self.model.parameters(), lr=lr,
                                 weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        criterion  = nn.CrossEntropyLoss()

        self.model.train()
        best_acc = 0.0

        for ep in range(epochs):
            total_loss, correct, total = 0.0, 0, 0

            for xb, yb in loader:
                logits = self.model(xb)          # (B, capacity)
                loss   = criterion(logits, yb)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)

            scheduler.step()
            acc = 100.0 * correct / total
            best_acc = max(best_acc, acc)
            print(f"  Epoch {ep+1:>3}/{epochs} | "
                  f"Loss: {total_loss/total:.4f} | "
                  f"Accuracy (vs OPT): {acc:.1f}%")

        print(f"\n  Best accuracy matching OPT: {best_acc:.1f}%")
        self.model.eval()
        self._save()
        self._trained = True

    # ── inference ─────────────────────────────────────────────────────
    def _evict_action(self, tracker: FeatureTracker,
                      cache: list[int]) -> int:
        state = tracker.cache_state(cache)          # (capacity, FEATURE_DIM)
        x     = torch.FloatTensor(state).unsqueeze(0)  # (1, capacity, FEATURE_DIM)
        with torch.no_grad():
            logits = self.model(x).squeeze(0)       # (capacity,)
        return int(logits.argmax().item())

    def request_sequence(self, sequence: list[int],
                         capacity: int | None = None) -> int:
        """
        Run the trained model on `sequence` and return page fault count.

        Parameters
        ----------
        sequence : ordered list of page references (ints in [0, max_pages))
        capacity : override cache size (defaults to self.capacity)

        Returns
        -------
        int — total page faults
        """
        cap     = capacity if capacity is not None else self.capacity
        cache   : list[int] = []
        tracker = FeatureTracker(self.max_pages)
        faults  = 0

        for page in sequence:
            tracker.update(page)

            if page in cache:
                continue   # hit

            faults += 1    # miss

            if len(cache) < cap:
                cache.append(page)
            else:
                idx        = self._evict_action(tracker, cache)
                cache[idx] = page

        return faults


# ─────────────────────────────────────────────────────────────────────────────
# Demo / benchmark
# ─────────────────────────────────────────────────────────────────────────────
def _benchmark(cache: ILCache, label: str, trace_fn, n: int = 5):
    il_total, lru_total, opt_total = 0, 0, 0
    for _ in range(n):
        ref       = trace_fn()
        il_total  += cache.request_sequence(ref)
        lru_total += lru_faults(ref, cache.capacity)
        opt_total += opt_faults(ref, cache.capacity)

    def pct(x): return f"{100*x/opt_total:.1f}% of OPT"

    print(f"\n  [{label}]  (avg over {n} traces)")
    print(f"    OPT : {opt_total//n:>6} faults")
    print(f"    IL  : {il_total//n:>6} faults  {pct(il_total)}")
    print(f"    LRU : {lru_total//n:>6} faults  {pct(lru_total)}")


def main():
    MAX_PAGES = 200
    CAPACITY  = 10

    print("=" * 60)
    print("  IL Page Replacement  —  Imitation of Bélády's OPT")
    print("=" * 60)

    cache = ILCache(max_pages=MAX_PAGES, capacity=CAPACITY)
    cache.train(n_traces=60, epochs=100)

    print("\n" + "=" * 60)
    print("  BENCHMARK  (fresh traces, never seen during training)")
    print("=" * 60)

    _benchmark(cache, "Working-set",
               lambda: _working_set_trace(MAX_PAGES))
    _benchmark(cache, "Zipf",
               lambda: _zipf_trace(MAX_PAGES))
    _benchmark(cache, "Looping (hard for LRU)",
               lambda: _looping_trace(MAX_PAGES, capacity=CAPACITY))

    # Demonstrate request_sequence API
    print("\n" + "=" * 60)
    custom = [random.randint(0, MAX_PAGES - 1) for _ in range(2000)]
    print(f"  request_sequence([2000 random refs]) → "
          f"{cache.request_sequence(custom)} faults")
    print("=" * 60)


if __name__ == "__main__":
    main()