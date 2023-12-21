import numpy as np
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_file_to_matrix(filename):
    assert os.path.isfile(filename), "File not found: %s" % filename

    camP = np.zeros((3, 4), dtype=np.float32)
    with open(filename, 'r') as mfile:
        lines = mfile.readlines()
        assert len(lines) == 3, str(lines)
        for r in range(len(lines)):
            words = lines[r].split(' ')
            assert len(words) == 4, str(words)
            for c in range(len(words)):
                camP[r, c] = float(words[c])

    return camP