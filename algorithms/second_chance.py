from __future__ import annotations


class SecondChancePageReplacement:
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.frames = []
        self.pointer = 0
        
        # Metrics
        self._count_page_faults = 0
        self.hits = 0

    def access_page(self, page):
        # Check if page exists
        for i in range(len(self.frames)):
            if self.frames[i][0] == page:
                # HIT → set reference bit = 1
                self.frames[i] = (page, 1)
                self.hits += 1
                return "HIT"

        # MISS
        self._count_page_faults += 1

        if len(self.frames) < self.capacity:
            self.frames.append((page, 1))
        else:
            while True:
                current_page, ref_bit = self.frames[self.pointer]

                if ref_bit == 0:
                  
                    self.frames[self.pointer] = (page, 1)
                    self.pointer = (self.pointer + 1) % self.capacity
                    break
                else:
                    
                    self.frames[self.pointer] = (current_page, 0)
                    self.pointer = (self.pointer + 1) % self.capacity

        return "MISS"

    def request_sequence(self, reference_string):
        history = []
        for page in reference_string:
            result = self.access_page(page)
            history.append((page, list(self.frames), result))

    @property
    def page_faults(self) -> int:
        return self._count_page_faults

if __name__ == "__main__":
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    capacity = 3

    sc = SecondChancePageReplacement(capacity)
    sc_result = sc.run(reference_string)



    print("Second Chance Results:")
    print("Page Faults:", sc_result["page_faults"])
    print("Hits:", sc_result["hits"])
    print("Miss Ratio:", sc_result["miss_ratio"])
    print("Hit Ratio:", sc_result["hit_ratio"])
    print("Execution Time:", sc_result["execution_time_sec"], "sec")
