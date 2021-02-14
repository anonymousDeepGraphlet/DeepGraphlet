import numpy as np
class Alias:
    def __init__(self, probs):
        self.K = len(probs)
        self.q = np.zeros(self.K)
        self.J = np.zeros(self.K, dtype=np.int)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger  = []
        for kk, prob in enumerate(probs):
            self.q[kk] = self.K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = self.q[large] - (1.0 - self.q[small])

            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw(self):
#         K  = len(J)

        # Draw from the overall uniform mixture.
        kk = int(np.floor(np.random.rand() * self.K))

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if np.random.rand() < self.q[kk]:
            return kk
        else:
            return self.J[kk]
if __name__ == "__main__":
    alias = Alias([0.1, 0.2, 0.1, 0.6])
    for i in range(10):
        print(alias.draw())