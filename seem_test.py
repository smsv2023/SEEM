# created 05/14/23 to test the SEEM models
# based on app.py: https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once/blob/main/demo_code/app.py

import os
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import torchvision.transforms as transforms
import torch
import argparse
import numpy as np
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from tasks import *
from inference_panoptic import infer_image_panoptic

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
print("loading model...")
cur_model = 'None'
if 'focalt' in args.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in args.conf_files:
    pretrained_pth = os.path.join("seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
print("building model...")
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

@torch.no_grad()
def inference(image, task, *args, **kwargs):
    audio = None
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return infer_image_panoptic(model, audio, image)
#        if 'Video' in task:
#            return interactive_infer_video(model, audio, image, task, *args, **kwargs)
#        else:
#            return interactive_infer_image(model, audio, image, task, *args, **kwargs)

print("opening test image...")
test_image_file = '/home/ec2-user/SAM/Segment-Everything-Everywhere-All-At-Once/demo_code/testset/Test010.png'
test_image=Image.open(test_image_file)

# Convert the image to RGB mode
test_image = test_image.convert("RGB")

# Convert the image to a NumPy array
image_array = np.array(test_image)

# Normalize the color values to the 0-1 range
image_array = image_array / 255.0

# Convert the normalized array back to a PIL Image object
test_image = Image.fromarray((image_array * 255).astype(np.uint8))

# Create the image dictionary
image_dict = {'image': test_image, "mask":None}

image_dict = {'image': test_image, "mask":None}

image_dict = {'image': test_image, "mask":None}

# Now you can feed the image into your model
print("start inferencing...")
result_image = inference(image_dict, task='Panoptic')[0]
print("saving result image...")
output_file = os.path.splitext(os.path.basename(test_image_file))[0]+"_result.png"
result_image.save(output_file, format='PNG')
