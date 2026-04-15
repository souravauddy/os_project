from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Final


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_PATH: Final[str] = os.path.join(CHECKPOINT_DIR, "dqn_model.pth")


# =========================
# WORKING SET TRACE
# =========================
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


# =========================
# CACHE ENV
# =========================
class CacheEnv:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.cache = []
        self.time = 0
        self.last_used = {}
        self.freq = {}
        return self.get_state()

    def get_state(self):
        state = []

        for p in self.cache:
            # rec = self.time - self.last_used.get(p, self.time)
            # freq = self.freq.get(p, 0)
            # state.extend([p, rec, freq])

            rec = (self.time - self.last_used.get(p, self.time)) / 100.0
            freq = self.freq.get(p, 0) / 10.0
            page_norm = p / 200.0

            state.extend([page_norm, rec, freq])

        while len(state) < self.capacity * 3:
            state.extend([0, 0, 0])

        return np.array(state, dtype=np.float32)

    def step(self, page, action):
        self.time += 1

        hit = page in self.cache

        if hit:
            self.last_used[page] = self.time
            self.freq[page] = self.freq.get(page, 0) + 1
            return self.get_state(), 1.0, False

        if len(self.cache) < self.capacity:
            self.cache.append(page)
        else:
            action = max(0, min(action, self.capacity - 1))
            self.cache[action] = page

        self.last_used[page] = self.time
        self.freq[page] = self.freq.get(page, 0) + 1

        return self.get_state(), -0.2, False


# =========================
# LRU BASELINE
# =========================
class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []

    def access(self, page):
        if page in self.cache:
            self.cache.remove(page)
            self.cache.append(page)
            return True
        else:
            if len(self.cache) >= self.capacity:
                self.cache.pop(0)
            self.cache.append(page)
            return False


# =========================
# ITEM-WISE Q NETWORK (FIX)
# =========================
class ItemQNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# REPLAY BUFFER
# =========================
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2 = zip(*batch)

        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(s2)
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# ACTION SELECTION (CRITICAL)
# =========================
def select_action(model, state, capacity, epsilon):
    if random.random() < epsilon:
        return random.randint(0, capacity - 1)

    state = state.reshape(capacity, 3)

    with torch.no_grad():
        scores = model(torch.FloatTensor(state)).squeeze()

    return torch.argmax(scores).item()


# =========================
# TRAINING
# =========================
def train(reference, capacity=64, episodes=100):

    model = ItemQNetwork()
    target = ItemQNetwork()
    target.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0

    for ep in range(episodes):

        env = CacheEnv(capacity)
        state = env.reset()

        total_reward = 0

        for step, page in enumerate(reference):

            action = select_action(model, state, capacity, epsilon)
            next_state, reward, _ = env.step(page, action)

            buffer.add(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(buffer) > 128:
                s, a, r, s2 = buffer.sample(64)

                s = torch.FloatTensor(s).reshape(-1, capacity, 3)
                s2 = torch.FloatTensor(s2).reshape(-1, capacity, 3)

                r = torch.FloatTensor(r)
                a = torch.LongTensor(a)

                q_values = model(s).squeeze(-1)
                q = q_values.gather(1, a.unsqueeze(1)).squeeze()

                next_q = target(s2).squeeze(-1)
                next_q_min = next_q.max(1)[0].detach()

                target_q = r + gamma * next_q_min

                loss = (q - target_q).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % 300 == 0:
                target.load_state_dict(model.state_dict())

        epsilon = max(0.05, epsilon * 0.98)

        print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")
    return model


# =========================
# EVALUATION
# =========================
def evaluate_dqn(model, reference, capacity):
    env = CacheEnv(capacity)
    state = env.reset()
    hits = 0

    for page in reference:
        state_reshaped = state.reshape(capacity, 3)

        with torch.no_grad():
            scores = model(torch.FloatTensor(state_reshaped)).squeeze()

        action = torch.argmax(scores).item()
        state, reward, _ = env.step(page, action)

        if reward > 0:
            hits += 1

    return hits / len(reference)


def evaluate_lru(reference, capacity):
    lru = LRU(capacity)
    hits = 0

    for p in reference:
        if lru.access(p):
            hits += 1

    return hits / len(reference)


# =========================
# MAIN
# =========================
def main():
    capacity = 64
    reference = generate_working_set_trace()

    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        model = ItemQNetwork()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    else:
        print("Training model...")
        model = train(reference, capacity=capacity, episodes=10)

    print("\nEvaluating...\n")

    dqn = evaluate_dqn(model, reference, capacity)
    lru = evaluate_lru(reference, capacity)

    print(f"DQN Hit Rate: {dqn:.4f}")
    print(f"LRU Hit Rate: {lru:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())