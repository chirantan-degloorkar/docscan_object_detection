import json

def process_darknet_output(output_json_path):
    '''
    Input - JSON file
    Output - Missing objects
    Function - Finds missing objects from the JSON and returns the objects
    '''
    try:
        with open(output_json_path, 'r') as file:
            data = json.load(file)
        
        detected_objects = set()
        for frame in data:
            for obj in frame.get('objects', []):
                detected_objects.add(obj['name'])
        
        print(f"Detected objects: {detected_objects}")

        all_objects = {'north-sign', 'color-stamp', 'detail-labels', 'bar-scale'}
        
        missing_objects = all_objects - detected_objects
        print(f"Missing objects: {missing_objects}")
        
        return missing_objects
    except Exception as e:
        print(f"Error processing Darknet output JSON: {e}")
        return set()
