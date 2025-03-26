import os, json
from PIL import Image, ImageDraw

# via_json_path = 'Image_Annotation/CL_JHCL_sS_SEG.json'; mask_output_dir = 'Masks/gt_sS_SEG/'
# via_json_path = 'Image_Annotation/CZ_sS_SEG.json'; mask_output_dir = 'Masks/gt_sS_SEG/'
# via_json_path = 'Image_Annotation/CL_JHCL_dS_DET.json'; mask_output_dir = 'Masks/gt_dS_DET/'
via_json_path = 'Image_Annotation/CZ_dS_DET.json'; mask_output_dir = 'Masks/gt_dS_DET/'

with open(via_json_path, 'r') as file:
    data = json.load(file)

os.makedirs(mask_output_dir, exist_ok=True)

# hard code dimension so as no need to provide image
image_width = 1936  # Adjust as needed
image_height = 1288  # Adjust as needed


classdata=[]
    
for key, value in data['_via_img_metadata'].items():
    # key, value = next(iter(data['_via_img_metadata'].items()))
    print(key)
    
    mask_id=255

    regions = value['regions']
    
    # Create a blank image with the same dimensions
    mask_image = Image.new('L', (image_width, image_height), 0)
    #print(np.histogram(mask_image, bins=np.append(np.unique(mask_image), np.inf)))

    draw = ImageDraw.Draw(mask_image)

    x_points = []
    y_points = []
    for region in regions:
        shape=region['shape_attributes']
        if 'all_points_x' in shape:
            # polygon
            x_points = shape['all_points_x']
            y_points = shape['all_points_y']
        elif shape['name']=='rect':
            # rectangle
            x_points = [shape['x'], shape['x']+shape['width'], shape['x']+shape['width'], shape['x']]
            y_points = [shape['y'], shape['y'], shape['y']+shape['height'], shape['y']+shape['height']]

        points = list(zip(x_points, y_points))
    
        # Draw the polygon with instance indices
        draw.polygon(points, outline=mask_id, fill=mask_id) 
        
        # # save key-mask_id-class
        # classlabel=region['region_attributes']['names']
        # classdata.append([key, mask_id, classlabel])
        
        mask_id=mask_id-1
    

    # Save the mask image
    mask_image.save(mask_output_dir+value['filename'].replace(".png","_mask.png"))
    

# deprecated
# Write class info to file

# Step 1: Extract the third item from each inner list
third_items = [item[2] for item in classdata]

# Step 2: Find unique third items and create a mapping of third item to unique index:
unique_labels = []
if via_json_path.__contains__('dS'):
    unique_labels = [
     'HSV1_pos/HSV2_neg',
     'HSV1_neg/HSV2_pos',
     'HSV1_pos/HSV2_pos',
     'HSV1_neg/HSV2_neg',
     'RPT/RPT',
     'R40/R40',
     'IND/IND',
     'HSV1_neg/IND',
     'HSV1_pos/IND',
     'IND/HSV2_neg'
     ]
elif via_json_path.__contains__('sS'):
    unique_labels = [
     'HSV1_neg',
     'HSV1_pos',
     'HSV2_neg',
     'HSV2_pos',
     'RPT',
     'R40',
     'IND'
     ]

    
    
unique_index_map = {value: index+1 for index, value in enumerate(unique_labels)}

# # Step 3: Add the index corresponding to the third item in each inner list
# for inner_list in classdata:
#     third_item = inner_list[2]
#     inner_list.append(unique_index_map[third_item])


# # Save class info to file
# with open(via_json_path.replace('.json','_json_maskIDclassID.csv'), 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['file_name', 'mask_id', 'class', 'class_id'])
#     writer.writerows(classdata)

