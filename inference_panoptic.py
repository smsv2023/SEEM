# created 05/14/23 for inferencing
# from original https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/blob/main/demo_code/tasks/interactive.py

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from xdecoder.language.loss import vl_similarity
#from utils.constants import COCO_PANOPTIC_CLASSES
#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

import cv2
import os
import glob
import subprocess
from PIL import Image
import random

t = []
t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
transform = transforms.Compose(t)
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

def infer_image_panoptic(model, audio_model, image):
    image_ori = transform(image['image'])
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    data = {"image": images, "height": height, "width": width}

    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    # processing Panoptic mode
    batch_inputs = [data]

    model.model.metadata = metadata
    results = model.model.evaluate(batch_inputs)
    pano_seg = results[-1]['panoptic_seg'][0]
    pano_seg_info = results[-1]['panoptic_seg'][1]
    demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
    res = demo.get_image()
    return Image.fromarray(res), pano_seg, pano_seg_info
