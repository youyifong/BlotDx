import json, math
import numpy as np

input_path1 = 'ImageLabels/CL_JHCL_singlestrip_IS.json'
input_path2 = 'ImageLabels/CL_JHCL_doublestrips_OD.json'

with open(input_path1, 'r') as file:
    singlestrip_in = json.load(file)['_via_img_metadata']

with open(input_path2, 'r') as file:
    doubstrip_in = json.load(file)['_via_img_metadata']


# change the keys in the two dict so that they can be matched
singlestrip_in = {key.split('.png')[0] + '.png': value for key, value in singlestrip_in.items()}
doubstrip_in = {key.split('.png')[0] + '.png': value for key, value in doubstrip_in.items()}


# hard code dimension so as no need to provide image
image_width = 1936  # Adjust as needed
image_height = 1288  # Adjust as needed


# find the closest regions among ss_regions to center, both from left and from right 
def find_left_right (center):
    dist=[]
    for region in ss_regions:
        shape=region['shape_attributes']
        center_x = np.mean(shape['all_points_x'])
        center_y = np.mean(shape['all_points_y'])
        # neg if from left and pos if from right
        dist.append((1 if center_x > center[0] else -1) * math.sqrt((center_x - center[0])**2 + (center_y - center[1])**2) )
    # find the closest from left
    left = np.argmin([np.Inf if x>0 else -x for x in dist])
    # find the closest from right
    right = np.argmin([np.Inf if x<0 else x for x in dist])
    
    return ((left, right))

flag=False        
    
for key, value in doubstrip_in.items():
    # key, value = next(iter(doubstrip_in.items()))
    #print(key)
        
    ds_regions = value['regions']    
    ss_regions = singlestrip_in[key]['regions']
        
    for region in ds_regions:
        shape=region['shape_attributes']
        center_x = shape['x']+shape['width']/2
        center_y = shape['y']+shape['height']/2
        left, right = find_left_right((center_x, center_y))
        
        if left==right:
            flag=True
            print("left==right")
            exit(-1)
            
        if ss_regions[left]['region_attributes']['names'][:4] == ss_regions[right]['region_attributes']['names'][:4] and ss_regions[left]['region_attributes']['names'].startswith('HSV'):
            print(key, left+1, ss_regions[left]['region_attributes']['names'], right+1, ss_regions[right]['region_attributes']['names'])
                
        region['region_attributes']['names'] = ss_regions[left]['region_attributes']['names'] + '/' + ss_regions[right]['region_attributes']['names']
        
    if flag:
        break


# write out
with open(input_path2, 'r') as file:
    doubstrip_out = json.load(file)
doubstrip_out['_via_img_metadata'] = doubstrip_in

doubstrip_out['_via_image_id_list'] = [x.split('.png')[0] + '.png' for x in doubstrip_out['_via_image_id_list']]

with open(input_path2.replace('.json', '1.json'), 'w') as f:
    json.dump(doubstrip_out, f)
    
# if the file looks good, replace CL_JHCL_doublestrips_OD.json with CL_JHCL_doublestrips_OD1.json