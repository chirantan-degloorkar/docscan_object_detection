import json
from collections import defaultdict

def create_ground_truth_dict(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of image_id to file_name
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # Use defaultdict to automatically initialize empty dictionaries for new keys
    ground_truth = defaultdict(lambda: {'boxes': [], 'labels': []})
    
    # Process annotations
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        filename = image_id_to_filename[image_id]
        
        # Extract bounding box coordinates
        x, y, width, height = annotation['bbox']
        box = [x, y, x + width, y + height]
        
        # Add box and label to the ground_truth dictionary
        ground_truth[filename]['boxes'].append(box)
        ground_truth[filename]['labels'].append(annotation['category_id'])
    
    # Convert defaultdict back to regular dict for final output
    return dict(ground_truth)

# Usage
# json_file_path = 'path/to/your/result.json'
# ground_truth = create_ground_truth_dict(json_file_path)

# Print the result (optional)
# print(json.dumps(ground_truth, indent=2))