import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

model = None

def prepare(args):
    global model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

def run(img1_path, img2_path, iters=20):
    img1 = torch.from_numpy(np.array(Image.open(img1_path)).astype(np.uint8)).permute(2, 0, 1).float()
    img2 = torch.from_numpy(np.array(Image.open(img2_path)).astype(np.uint8)).permute(2, 0, 1).float()
    images = torch.stack([img1, img2], dim=0).to(DEVICE)
    images = InputPadder(images.shape).pad(images)[0]

    with torch.no_grad():
        image1 = images[0, None]
        image2 = images[1, None]

        flow_low, flow_up = model(image1, image2, iters=iters, test_mode=True)
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()

    return flow
