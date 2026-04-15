import os
import random
import pickle
from collections import deque


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lpr_model.pkl")


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []
    
    def access(self, page):
        if page in self.cache:
            self.cache.remove(page)
            self.cache.append(page)
            return False  # hit
        return True  # fault
    
    def victim(self):
        return self.cache[0]  # least recent


class MRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []
    
    def access(self, page):
        if page in self.cache:
            self.cache.remove(page)
            self.cache.append(page)
            return False
        return True
    
    def victim(self):
        return self.cache[-1]  # most recent


class LPRModel:
    def __init__(self, memory_size, lr=0.3):
        self.memory_size = memory_size
        self.lr = lr
        
        self.w_lru = 0.5
        self.w_mru = 0.5
        
        self.discount = 0.005 ** (1 / memory_size)
        
        # eviction history: page -> (time, policy)
        self.history = {}
        self.time = 0

    def choose_policy(self):
        # exploration
        exploration = random.random()
        
        if exploration < 0.1:  # 10% exploration
            return random.choice(["LRU", "MRU"])
        
        return "LRU" if random.random() < self.w_lru else "MRU"

    def update(self, page):
        if page not in self.history:
            return
        
        evict_time, policy = self.history[page]
        t = self.time - evict_time
        
        reward = self.discount ** t
        
        if policy == "LRU":
            self.w_mru += self.lr * reward
            self.w_lru -= self.lr * reward
        else:
            self.w_lru += self.lr * reward
            self.w_mru -= self.lr * reward
        
        # normalize
        total = self.w_lru + self.w_mru
        self.w_lru /= total
        self.w_mru /= total

    def record_eviction(self, page, policy):
        self.history[page] = (self.time, policy)


class LPRSimulator:
    def __init__(self, model, capacity):
        self.model = model
        self.capacity = capacity
        
        self.cache = []
        self.page_faults = 0

    def run(self, reference_string):
        for page in reference_string:
            self.model.time += 1
            
            # hit
            if page in self.cache:
                self.model.update(page)
                self.cache.remove(page)
                self.cache.append(page)
                continue
            
            # fault
            self.page_faults += 1
            
            if len(self.cache) < self.capacity:
                self.cache.append(page)
                continue
            
            # simulate both policies
            victim_lru = self.cache[0]
            victim_mru = self.cache[-1]
            
            policy = self.model.choose_policy()
            
            if policy == "LRU":
                victim = victim_lru
            else:
                victim = victim_mru
            
            # record eviction
            self.model.record_eviction(victim, policy)
            
            # replace
            self.cache.remove(victim)
            self.cache.append(page)
        
        return self.page_faults


def generate_synthetic_trace(length=10000, num_pages=50):
    trace = []
    
    for _ in range(length):
        r = random.random()
        
        if r < 0.5:
            # locality
            trace.append(random.randint(0, 10))
        elif r < 0.8:
            # sequential pattern
            trace.append(_ % num_pages)
        else:
            # random
            trace.append(random.randint(0, num_pages - 1))
    
    return trace


def train_model(memory_size=64, episodes=150):
    model = LPRModel(memory_size)
    
    for ep in range(episodes):
        trace = generate_synthetic_trace()
        sim = LPRSimulator(model, memory_size)
        faults = sim.run(trace)
        print(f"Episode {ep+1}, Faults: {faults}")
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved.")

    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("Train model first!")
    
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_faults(max_pages, reference_string):
    model = load_model()
    
    model.history = {}
    model.time = 0
    
    sim = LPRSimulator(model, max_pages)
    faults = sim.run(reference_string)
    
    return faults


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        train_model(memory_size=64, episodes=150)
    
    ref = [1,2,3,4,1,2,5,1,2,3,4,5]
    faults = predict_faults(3, ref)

    print("RL Page Faults:", faults)
