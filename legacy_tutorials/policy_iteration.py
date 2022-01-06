import numpy as np

# Initialize
v_s = [[0, 0, 0], [0, None, 0], [0, 0, 0]]

for v in zip(np.arange(3), np.arange(3)):
    print(v)

def transitions(pos):
    if pos == [0, 0]:
        return
