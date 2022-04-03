from collections import Counter

import numpy as np

print([i for i in range(10)])

counter = Counter()
counter['what'] += 0.37

print(counter)

array = np.zeros((10_000, 3))

print(array.itemsize * array.size)