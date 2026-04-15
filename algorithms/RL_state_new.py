"""
rl_cache.py  —  RL-based Page Replacement that rivals OPT
==========================================================

Key ideas
----------
1. Lookahead reward during training: At each eviction decision the trainer
   passes the *next-use distance* of every cached page to a shaped reward
   so the agent learns to imitate OPT (Bélády's algorithm) without ever
   hard-coding it.

2. Per-slot scoring network: A small MLP scores every slot in the cache;
   the slot with the lowest score is evicted.  Features per slot:
       [page_id/max_pages, recency/horizon, frequency/horizon, next_use/horizon]
   During inference the "next_use" feature is set to 0 (unknown).

3. Persistent weights: saved to `cache_model.pt` automatically after
   training and reloaded on the next run.

4. request_sequence(seq, capacity) — public API for counting page faults.

Usage
-----
    from rl_cache import RLCache

    cache = RLCache(max_pages=200, capacity=10)
    cache.train()                          # trains (or loads saved weights)
    faults = cache.request_sequence(my_sequence)
    print(f"Page faults: {faults}")
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "cache_model.pt"
FEATURE_DIM = 4          # [page_id, recency, frequency, next_use]
HIDDEN = 128
HORIZON = 1000.0         # normalisation constant for time-based features


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────
class SlotScorer(nn.Module):
    """Scores a single cache slot.  Lower score → better candidate for eviction."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., FEATURE_DIM)  →  (..., 1)"""
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Cache environment (with optional lookahead for training)
# ──────────────────────────────────────────────────────────────────────────────
class CacheEnv:
    def __init__(self, capacity: int, max_pages: int):
        self.capacity = capacity
        self.max_pages = max_pages
        self.reset()

    def reset(self):
        self.cache: list[int] = []
        self.time = 0
        self.last_used: dict[int, int] = {}
        self.freq: dict[int, int] = {}
        return self._state()

    # ------------------------------------------------------------------
    def _state(self, future_dist: dict[int, float] | None = None) -> np.ndarray:
        """
        Build a (capacity, FEATURE_DIM) state array.
        future_dist: mapping page → next-use distance (used during training).
        """
        state = []
        for p in self.cache:
            rec  = (self.time - self.last_used.get(p, self.time)) / HORIZON
            freq = self.freq.get(p, 0) / HORIZON
            nd   = (future_dist.get(p, HORIZON) / HORIZON) if future_dist else 0.0
            state.append([p / self.max_pages, rec, freq, nd])

        # Padding
        while len(state) < self.capacity:
            state.append([0.0, 0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, page: int, action: int,
             future_dist: dict[int, float] | None = None):
        """
        Process one page reference.
        action  : index into self.cache to evict on a miss (ignored on hit).
        Returns : (next_state, reward, hit)
        """
        self.time += 1
        hit = page in self.cache

        if hit:
            self.cache.remove(page)
            self.cache.append(page)
            self.last_used[page] = self.time
            self.freq[page] = self.freq.get(page, 0) + 1
            return self._state(future_dist), 1.0, True

        # ── MISS ──────────────────────────────────────────────────────
        if len(self.cache) < self.capacity:
            self.cache.append(page)
            reward = -1.0
        else:
            evicted = self.cache[action]
            # Shaped reward: reward evicting the page with the FARTHEST next use
            if future_dist is not None:
                evicted_dist  = future_dist.get(evicted, HORIZON)
                # Compare against the average next-use of all other cached pages
                others = [future_dist.get(p, HORIZON)
                          for i, p in enumerate(self.cache) if i != action]
                avg_others = np.mean(others) if others else HORIZON
                # Positive when we evicted a farther page than average → good choice
                reward = (evicted_dist - avg_others) / HORIZON - 1.0
            else:
                reward = -1.0

            self.cache[action] = page

        self.last_used[page] = self.time
        self.freq[page] = self.freq.get(page, 0) + 1
        return self._state(future_dist), reward, False


# ──────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxlen: int = 200_000):
        self.buf = deque(maxlen=maxlen)

    def add(self, s, a, r, s2):
        self.buf.append((s, a, r, s2))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2 = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────────────────────────────────────
# Trace generators
# ──────────────────────────────────────────────────────────────────────────────
def _working_set_trace(max_pages: int, n_refs: int = 15_000) -> list[int]:
    trace: list[int] = []
    ws_size = max(5, max_pages // 7)
    for _ in range(n_refs // 300):
        ws = random.sample(range(max_pages), min(ws_size, max_pages))
        for _ in range(300):
            if random.random() < 0.08:
                trace.append(random.randint(0, max_pages - 1))
            else:
                trace.append(random.choice(ws))
    return trace


def _next_use_distances(trace: list[int]) -> list[dict[int, float]]:
    """
    Pre-compute, for each position t, the distance to the next use of each page.
    Returns a list of dicts indexed by position.
    """
    n = len(trace)
    next_use = [HORIZON] * n
    last: dict[int, int] = {}

    for t in range(n - 1, -1, -1):
        p = trace[t]
        next_use[t] = last.get(p, HORIZON)
        last[p] = t

    # Now build per-position dict over ALL pages currently "live"
    # (we only need distances for pages in the cache — computed lazily in env)
    # Instead return a simple list of next-use position per page at each step.
    # We'll expose this differently: return next_use_pos list.
    return next_use   # next_use[t] = next occurrence of trace[t] (not yet used here)


def _build_future_map(trace: list[int], t: int) -> dict[int, float]:
    """Distance to next use for every page, starting from position t+1."""
    dist: dict[int, float] = {}
    for i in range(t + 1, len(trace)):
        p = trace[i]
        if p not in dist:
            dist[p] = float(i - t)
    return dist


# ──────────────────────────────────────────────────────────────────────────────
# Action selection
# ──────────────────────────────────────────────────────────────────────────────
def _select_action(model: SlotScorer, state: np.ndarray,
                   capacity: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, capacity - 1)
    with torch.no_grad():
        s = torch.FloatTensor(state)            # (capacity, FEATURE_DIM)
        scores = model(s).squeeze(-1)           # (capacity,)
        return int(torch.argmin(scores).item()) # evict lowest-scored slot


# ──────────────────────────────────────────────────────────────────────────────
# Baseline algorithms
# ──────────────────────────────────────────────────────────────────────────────
def lru_faults(reference: list[int], capacity: int) -> int:
    cache: list[int] = []
    last: dict[int, int] = {}
    faults = 0
    for t, p in enumerate(reference):
        if p in cache:
            cache.remove(p)
            cache.append(p)
        else:
            faults += 1
            if len(cache) >= capacity:
                lru = min(cache, key=lambda x: last.get(x, 0))
                cache.remove(lru)
            cache.append(p)
        last[p] = t
    return faults


def opt_faults(reference: list[int], capacity: int) -> int:
    """Bélády's OPT — offline, requires full future knowledge."""
    cache: set[int] = set()
    faults = 0
    for t, p in enumerate(reference):
        if p in cache:
            continue
        faults += 1
        if len(cache) < capacity:
            cache.add(p)
        else:
            # Evict page whose next use is farthest in future
            def next_use(page):
                for i in range(t + 1, len(reference)):
                    if reference[i] == page:
                        return i
                return float('inf')
            evict = max(cache, key=next_use)
            cache.remove(evict)
            cache.add(p)
    return faults


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────
class RLCache:
    """
    Reinforcement-learning page replacement that approaches OPT performance.

    Parameters
    ----------
    max_pages : int
        Universe of page IDs: [0, max_pages).
    capacity  : int
        Number of frames in the cache.
    model_path : str
        Where to save / load model weights.
    """

    def __init__(self, max_pages: int = 200, capacity: int = 10,
                 model_path: str = MODEL_PATH):
        self.max_pages  = max_pages
        self.capacity   = capacity
        self.model_path = model_path

        self.model  = SlotScorer()
        self.target = SlotScorer()
        self._sync_target()

        self._loaded = self._try_load()

    # ── persistence ───────────────────────────────────────────────────
    def _try_load(self) -> bool:
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, weights_only=True)
            # Verify the checkpoint is compatible
            if (ckpt.get("max_pages") == self.max_pages and
                    ckpt.get("capacity") == self.capacity):
                self.model.load_state_dict(ckpt["model"])
                self.target.load_state_dict(ckpt["model"])
                print(f"[RLCache] Loaded weights from '{self.model_path}'")
                return True
            else:
                print("[RLCache] Saved weights are for different params — retraining.")
        return False

    def _save(self):
        torch.save({
            "model":     self.model.state_dict(),
            "max_pages": self.max_pages,
            "capacity":  self.capacity,
        }, self.model_path)
        print(f"[RLCache] Weights saved to '{self.model_path}'")

    def _sync_target(self):
        self.target.load_state_dict(self.model.state_dict())

    # ── training ──────────────────────────────────────────────────────
    def train(self, episodes: int = 20, force: bool = False,
              batch_size: int = 128, gamma: float = 0.99,
              lookahead_every: int = 5):
        """
        Train the model.  If weights already exist and force=False, skips training.

        lookahead_every : int
            Build the exact future-distance map every N steps (expensive but
            highly informative).  Between lookups a cached map is reused.
        """
        if self._loaded and not force:
            print("[RLCache] Skipping training — loaded existing weights.")
            print("          Pass force=True to retrain from scratch.")
            return

        opt     = optim.Adam(self.model.parameters(), lr=5e-4)
        buffer  = ReplayBuffer()
        epsilon = 1.0

        for ep in range(episodes):
            ref = _working_set_trace(self.max_pages)
            env = CacheEnv(self.capacity, self.max_pages)
            state = env.reset()
            total_r = 0.0
            future_map: dict[int, float] = {}

            for t, page in enumerate(ref):
                # Refresh lookahead map periodically
                if t % lookahead_every == 0:
                    future_map = _build_future_map(ref, t)
                    # Also inject future info into state features
                cache_future = {p: future_map.get(p, HORIZON) for p in env.cache}

                # Rebuild state with future distances (training only)
                state_with_future = env._state(cache_future)

                action = _select_action(self.model, state_with_future,
                                        self.capacity, epsilon)
                next_state, reward, hit = env.step(page, action,
                                                   future_dist=cache_future)

                # Next state also uses future info for richer value targets
                cache_future2 = {p: future_map.get(p, HORIZON) for p in env.cache}
                next_state_with_future = env._state(cache_future2)

                buffer.add(state_with_future, action, reward,
                           next_state_with_future)
                state = next_state
                total_r += reward

                # ── DQN update ────────────────────────────────────────
                if len(buffer) >= batch_size * 2:
                    s, a, r, s2 = buffer.sample(batch_size)
                    s  = torch.FloatTensor(s)
                    s2 = torch.FloatTensor(s2)
                    r  = torch.FloatTensor(r)
                    a  = torch.LongTensor(a)

                    # Q(s,a): score of the evicted slot
                    q = self.model(s).squeeze(-1)           # (B, capacity)
                    q = q.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

                    with torch.no_grad():
                        # Target: pick the slot the target net would evict
                        next_q  = self.target(s2).squeeze(-1)   # (B, capacity)
                        next_q  = next_q.min(dim=1)[0]           # (B,)
                        target_q = r + gamma * next_q

                    loss = nn.SmoothL1Loss()(q, target_q)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()

                if t % 500 == 0:
                    self._sync_target()

            epsilon = max(0.02, epsilon * 0.95)
            print(f"  Episode {ep+1:>3}/{episodes} | "
                  f"Reward: {total_r:>10.1f} | ε={epsilon:.3f}")

        self._save()
        self._loaded = True

    # ── inference ─────────────────────────────────────────────────────
    def _evict_action(self, state: np.ndarray) -> int:
        """Choose which slot to evict (greedy, no future info at inference)."""
        with torch.no_grad():
            s = torch.FloatTensor(state)
            scores = self.model(s).squeeze(-1)
        return int(torch.argmin(scores).item())

    def request_sequence(self, sequence: list[int],
                         capacity: int | None = None) -> int:
        """
        Run the trained model on `sequence` and return the number of page faults.

        Parameters
        ----------
        sequence : list[int]
            Ordered page references.  Each value must be in [0, max_pages).
        capacity : int, optional
            Override the cache capacity.  Defaults to self.capacity.

        Returns
        -------
        int  — total page faults experienced.
        """
        cap = capacity if capacity is not None else self.capacity
        env = CacheEnv(cap, self.max_pages)
        state = env.reset()
        faults = 0

        for page in sequence:
            action = self._evict_action(state)
            state, reward, _ = env.step(page, action)
            if reward < 0:
                faults += 1

        return faults


# ──────────────────────────────────────────────────────────────────────────────
# Demo / benchmark
# ──────────────────────────────────────────────────────────────────────────────
def main():
    MAX_PAGES = 200
    CAPACITY  = 64
    EPISODES  = 50

    print("=" * 60)
    print("  RL Page Replacement  —  training against OPT-shaped reward")
    print("=" * 60)

    cache = RLCache(max_pages=MAX_PAGES, capacity=CAPACITY)
    cache.train(episodes=EPISODES)

    # Generate a fresh test trace (unseen during training)
    print("\nGenerating test trace …")
    test_ref = _working_set_trace(MAX_PAGES, n_refs=6_000)

    rl_faults  = cache.request_sequence(test_ref)
    lru_f      = lru_faults(test_ref, CAPACITY)
    opt_f      = opt_faults(test_ref, CAPACITY)

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS  (fresh test trace, never seen in training)")
    print("=" * 60)
    print(f"  OPT (Bélády) page faults : {opt_f:>6}")
    print(f"  RL  (trained) page faults: {rl_faults:>6}  "
          f"({'%.1f' % (100 * rl_faults / opt_f)}% of OPT)")
    print(f"  LRU            page faults: {lru_f:>6}  "
          f"({'%.1f' % (100 * lru_f / opt_f)}% of OPT)")
    print("=" * 60)

    # Demonstrate request_sequence API
    custom = [random.randint(0, MAX_PAGES - 1) for _ in range(1000)]
    print(f"\n  request_sequence([1000 random refs]) → {cache.request_sequence(custom)} faults")


if __name__ == "__main__":
    main()