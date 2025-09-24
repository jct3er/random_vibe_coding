import sys
import numpy as np

class LCG:
    """
    Linear Congruential Generator (mixed form).
    Not cryptographically secure. Use for simulations/learning/demo only.
    """
    def __init__(self, seed: int = 1, a: int = 1664525, c: int = 1013904223, m: int = 2**32):
        if not (0 <= seed < m):
            raise ValueError("seed must satisfy 0 <= seed < m")
        self.a = int(a)
        self.c = int(c)
        self.m = int(m)
        self.state = int(seed)

    def next_int(self) -> int:
        """Return next integer in [0, m-1]."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self) -> float:
        """Return float in [0.0, 1.0)."""
        # divide by m (not m-1) to get values in [0, 1)
        return self.next_int() / self.m

    def randrange(self, start: int, stop: int = None) -> int:
        """
        Return a random integer in range(start, stop) like random.randrange.
        If stop is None, returns in [0, start).
        """
        if stop is None:
            low, high = 0, int(start)
        else:
            low, high = int(start), int(stop)
        if low >= high:
            raise ValueError("empty range")
        # uniformly map state to the interval length
        length = high - low
        # use rejection sampling to reduce bias when length does not divide m cleanly
        while True:
            x = self.next_int()
            # Accept x if it is within a largest multiple of length below m:
            limit = (self.m // length) * length
            if x < limit:
                return low + (x % length)
    
    def random_intarray(self, start: int, stop: int = None, size=1):
        """
        Return an array with random integers between start and stop 
        """

        return np.array([self.randrange(start=start, stop=stop) for i in range(size)])
    
    def random_array(self, low: float = 0.0, high: float = 1.0, n: int=1):
        """
        Generate an array of n random floats in [low, high).
        """
        if high <= low:
            raise ValueError("high must be greater than low")
        scale = high - low
        return np.array([low + scale * self.random() for _ in range(n)])

    def seed(self, new_seed: int):
        if not (0 <= new_seed < self.m):
            raise ValueError("seed must be 0 <= seed < m")
        self.state = int(new_seed)

    def __iter__(self):
        # infinite iterator of ints
        while True:
            yield self.next_int()

if __name__ == "__main__":
    r0=10000
    if len(sys.argv)>1: r0=int(sys.argv[1])
    rng = LCG(seed=r0)
    print(rng.next_int())     # integer in [0, 2**32-1]
    print(rng.random())       # float in [0.0, 1.0)
    print(rng.randrange(10))  # integer in [0,9]
    print(rng.randrange(5, 15)) # integer in [5,14]
    print(rng.random_intarray(50,150,1000))
    print(rng.random_array(50,150,1000))

