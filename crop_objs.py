from skimage.transform import resize

def save_cropped_obj_mask(obj, image_array):
    # Create a binary mask for this object
    print("Crop object", obj['id'])

    mask = (pano_seg == obj['id']).cpu().numpy().astype(np.uint8) * 255

    # Resize the mask to match the image_array dimensions
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
