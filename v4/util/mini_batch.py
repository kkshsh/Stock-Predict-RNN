import numpy as np

def batch(batch_size, data=None, target=None, softmax=None, shuffle=False):
    if target:
        assert len(data) == len(target)
    else:
        assert len(data) == len(softmax)

    if shuffle:
        indices = np.arange(len(data), dtype=np.int32)
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if target:
            yield data[excerpt], target[excerpt], softmax[excerpt]
        else:
            yield data[excerpt], None, softmax[excerpt]
