#from skimage.transform import resize
#from read_pfm import read_pfm
#from write_pfm import write_pfm
from PIL import Image
import numpy as np
import torch
import os
import cv2
import json

COCO_PANOPTIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

def save_cropped_obj_mask(obj, image_array, output_path):
    # Create a binary mask for this object
    print("Crop object", obj['id'])

    mask = (pano_seg == obj['id']).cpu().numpy().astype(np.uint8) * 255

    # Resize the mask to match the image_array dimensions
    #mask_resized = resize(mask, (image_array.shape[0], image_array.shape[1]))
    mask_resized = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Assuming 'mask' is your binary mask
    dilated_mask_resized = cv2.dilate(mask_resized, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    # Crop the original image using the mask
    cropped = image_array * np.expand_dims(dilated_mask_resized, axis=-1)

    # Save the mask and cropped image to files
    #cropped_image_path=os.path.join(output_path, os.path.splitext(basename)[0])
    os.makedirs(output_path, exist_ok=True)

    obj_name = str(obj['id']) + "_" + COCO_PANOPTIC_CLASSES[obj['category_id']]
    print("saving object: %s..."%obj_name)
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(output_path, f"{obj['id']}_{obj_name}_mask.png"))
    
    mask_img_resized = Image.fromarray(mask_resized)
    mask_img_resized.save(os.path.join(output_path, f"obj['id']}_{obj_name}_mask_resized.png"))
    
    cropped_img = Image.fromarray(cropped.astype(np.uint8))
    cropped_img.save(os.path.join(output_path, f"obj['id']}_{obj_name}_cropped.png"))        

# imageio has issues when access pfm create by MiDaS
def save_cropped_depth_map(obj, depth_map):
    # Create a binary mask for this object
    print("crop depth map of object", obj['id'])
    obj_name = str(obj['id']) + "_" + COCO_PANOPTIC_CLASSES[obj['category_id']]
    mask = (pano_seg == obj['id']).cpu().numpy().astype(np.uint8)

    # Resize the mask to match the depth_map dimensions
    #mask_resized = resize(mask, (depth_map.shape[0], depth_map.shape[1]))
    
    mask_resized = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
    
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
    
folder = '/Users/seanmao/Pictures/SEEM/testset/'
file = 'Test001.png'
image = Image.open(os.path.join(folder,file))
# Convert the image to RGB mode
image = image.convert("RGB")
# Convert the image to a NumPy array
image_array = np.array(image)

result_folder = '/Users/seanmao/Pictures/SEEM/output'
pth_file = 'Test001_result.pth'
# Load pano_seg tensor
pano_seg = torch.load(os.path.join(result_folder,pth_file), map_location=torch.device('cpu') )

json_file = 'Test001_result.json'
# Load pano_seg_info list of dictionaries
with open(os.path.join(result_folder,json_file), 'r') as f:
    pano_seg_info = json.load(f)

# if pfm
#depth_map_path = os.path.join(input_path_pfm, os.path.splitext(basename)[0]) + f"-dpt_swin2_large_384.pfm"
#depth_map = read_pfm(depth_map_path)

output_folder = os.path.join(result_folder, os.path.splitext(os.path.basename(file))[0])
for obj in pano_seg_info:
    # For each object in pano_seg_info, create a mask and crop the original image
    print ("creating cropped object and mask files...")
    save_cropped_obj_mask(obj, image_array, output_folder)
    #print ("creating cropped depth map...")
    #save_cropped_depth_map(obj, depth_map)    
