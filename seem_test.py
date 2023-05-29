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

# process a specified image file

def process_file(test_image_file):
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

   # Now you can feed the image into your model
   print("start inferencing...")
   #result_image, pano_seg, pano_seg_info = inference(image_dict, task='Panoptic')
   pano_seg, pano_seg_info = inference(image_dict, task='Panoptic')

   # Save image file with color coded pixels and object names
   #output_image_file = os.path.splitext(test_image_basename)[0]+"_result.png"
   #result_image.save(output_image_file, format='PNG')
   return pano_seg, pano_seg_info, image_array


# open the image file and process
#test_image_file = '/home/ec2-user/SAM/Segment-Everything-Everywhere-All-At-Once/demo_code/testset/Test001.png'
#process_file(test_image_file)

# get input
import glob
input_path = '/home/ec2-user/SAM/Segment-Everything-Everywhere-All-At-Once/demo_code/testset/'
input_path_pfm ='/home/ec2-user/midas/MiDaS/output'
if input_path is not None:
    image_names = glob.glob(os.path.join(input_path, "*.png"))
    num_images = len(image_names)
else:
    print("No input path specified.")

# create output folder
output_path = '/home/ec2-user/SAM/Segment-Everything-Everywhere-All-At-Once/demo_code/output/'
if output_path is not None:
    os.makedirs(output_path, exist_ok=True)
print("Start processing")

for index, image_name in enumerate(image_names):
    basename = os.path.basename(image_name)
    print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))
    # open the image file and process
    pano_seg, pano_seg_info, image_array = process_file(image_name)

    print("saving result files...")

    # Save pano_seg tensor
    output_pth_file = output_path + os.path.splitext(basename)[0]+"_result.pth"
    torch.save(pano_seg, output_pth_file)

    # Save pano_seg_info list of dictionaries
    #import pickle
    #output_pkl_file = output_path + os.path.splitext(basename)[0]+"_result.pkl"
    #with open(output_pkl_file, 'wb') as f:
    #   pickle.dump(pano_seg_info, f)

    import json
    output_json_file = output_path + os.path.splitext(basename)[0]+"_result.json"
    # Save pano_seg_info list of dictionaries
    with open(output_json_file, 'w') as f:
        json.dump(pano_seg_info, f)

    # Load pano_seg_info list of dictionaries
    #with open('pano_seg_info.json', 'r') as f:
    #    pano_seg_info = json.load(f)

    def save_cropped_obj_mask(obj, image_array):
        # Create a binary mask for this object
        print("Crop object", obj['id'])

        mask = (pano_seg == obj['id']).cpu().numpy().astype(np.uint8) * 255

        # Resize the mask to match the image_array dimensions
        from skimage.transform import resize
        mask_resized = resize(mask, (image_array.shape[0], image_array.shape[1]))

        # Crop the original image using the mask
        cropped = image_array * np.expand_dims(mask_resized, axis=-1)

        # Save the mask and cropped image to files
        cropped_image_path=os.path.join(output_path, os.path.splitext(basename)[0])
        os.makedirs(cropped_image_path, exist_ok=True)

        obj_name = str(obj['id']) + "_" + COCO_PANOPTIC_CLASSES[obj['category_id']]
        print("saving object: %s..."%obj_name)
        mask_img = Image.fromarray(mask)
        #mask_img.save(os.path.join(cropped_image_path, f"{obj['id']}_mask.png"))
        mask_img.save(os.path.join(cropped_image_path, f"{obj_name}_mask.png"))
        cropped_img = Image.fromarray(cropped.astype(np.uint8))
        cropped_img.save(os.path.join(cropped_image_path, f"{obj_name}_cropped.png"))        
  
    # imageio has issues when access pfm create by MiDaS
    from read_pfm import read_pfm
    from write_pfm import write_pfm
    def save_cropped_depth_map(obj, depth_map):
        # Create a binary mask for this object
        print("crop depth map of object", obj['id'])
        obj_name = str(obj['id']) + "_" + COCO_PANOPTIC_CLASSES[obj['category_id']]
        mask = (pano_seg == obj['id']).cpu().numpy().astype(np.uint8)
        
        # Resize the mask to match the depth_map dimensions
        from skimage.transform import resize
        mask_resized = resize(mask, (depth_map.shape[0], depth_map.shape[1]))
        # binarize the mask again, value between 0 and 1 can be introduced when resizing
        mask_resized = (mask_resized > 0.5).astype(np.float32) 
        
        # if png, expand dimensions of the mask to match the depth map
        #mask_resized = np.expand_dims(mask_resized, axis=-1)
        #mask_resized = np.repeat(mask_resized, 3, axis=-1)

        # Isolate the object in the depth map using the mask
        isolated_object_depth_map = depth_map * mask_resized
        
        # Save the isolated object depth map to a file
        # if png
        #isolated_object_depth_map_path = os.path.join(output_path, os.path.splitext(basename)[0], f"{obj_name}_isolated_object_depth_map.png")
        #isolated_object_depth_map_img = Image.fromarray(isolated_object_depth_map.astype(np.uint8))
        #isolated_object_depth_map_img.save(isolated_object_depth_map_path)
        
        # if pfm
        isolated_object_depth_map_path = os.path.join(output_path, os.path.splitext(basename)[0], f"{obj_name}_isolated_object_depth_map.pmf")
        
        # Normalize the depth values to [0, 1], original pfm is not nomalized, so don't do it. 
        # isolated_object_depth_map = (isolated_object_depth_map - np.min(isolated_object_depth_map)) / (np.max(isolated_object_depth_map) - np.min(isolated_object_depth_map))

        # Convert to float32
        isolated_object_depth_map = isolated_object_depth_map.astype(np.float32)

        # Save the isolated object depth map to a file
        write_pfm(isolated_object_depth_map_path, isolated_object_depth_map)

    
    # scale the color values from the 0-1 range to 0-255 range
    image_array = image_array * 255
    
    # Load depth map
    # if png
    #depth_map_path = os.path.join(os.path.splitext(image_name)[0], f"-dpt_swin2_large_384.png")
    #depth_map_img = PIL.Image.open(depth_map_path)
    #depth_map = np.array(depth_map_img)
    
    # if pfm
    depth_map_path = os.path.join(input_path_pfm, os.path.splitext(basename)[0]) + f"-dpt_swin2_large_384.pfm"
    depth_map = read_pfm(depth_map_path)
  
    for obj in pano_seg_info:
        # For each object in pano_seg_info, create a mask and crop the original image
        print ("creating cropped object and mask files...")
        save_cropped_obj_mask(obj, image_array)
        print ("creating cropped depth map...")
        save_cropped_depth_map(obj, depth_map)
