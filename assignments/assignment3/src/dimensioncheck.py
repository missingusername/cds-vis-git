import os
import tqdm
from PIL import Image
import pandas as pd

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

def get_image_dimensions(folder_path):
    dimensions_count = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more image extensions if needed
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    dimension = f"{width}x{height}"
                    if dimension in dimensions_count:
                        dimensions_count[dimension] += 1
                    else:
                        dimensions_count[dimension] = 1
    return dimensions_count

def create_dataframe(dimensions_count):
    df = pd.DataFrame(list(dimensions_count.items()), columns=['dimension', 'count'])
    return df

# Specify the folder path containing the images
folder_path = os.path.join('..','in','Tobacco3482-jpg')

# Get dimensions count
dimensions_count = get_image_dimensions(folder_path)

# Create DataFrame
df = create_dataframe(dimensions_count)

# Display DataFrame
print(df)
