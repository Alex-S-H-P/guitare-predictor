class Normalizer:

    def __init__(self):
        self.min_data: float = +float("inf")
        self.max_data: float = -float("inf")

    def fit_add(self, x: float):
        if x < self.min_data:
            self.min_data = x
        if x > self.max_data:
            self.max_data = x

    def __call__(self, x: float, /) -> float:
        """Normalizes x in the data or at least in a similarly distributed space

        :arg x a floating point value
        """
        return (x - self.min_data) / (self.max_data - self.min_data)
