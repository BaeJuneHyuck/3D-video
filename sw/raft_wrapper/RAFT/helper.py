import wrapper
import numpy as np
from argparse import Namespace

wrapper.prepare(Namespace(model='models/raft-things.pth', small=False, mixed_precision=False))

def run(image1_path = None, image2_path = None):
    return wrapper.run(image1_path, image2_path, iters=20)
