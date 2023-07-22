import os
import json

# Create a directory only if it doesn't already exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Gets the file extension of a file
def get_file_extension(source_file):
    return os.path.splitext(source_file)[1].lower()

# Append the information to the JSON Object
def append_to_json_file(file_path, new_objects):
    # Read the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Append new objects to the existing data
    data.append(new_objects)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)