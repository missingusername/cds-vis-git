import os
import argparse
import pandas as pd
import cv2
from facenet_pytorch import MTCNN
from tqdm import tqdm
import time

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

data_path = os.path.join('..','in')

# global Initialization of MTCNN for face detection
mtcnn = MTCNN(keep_all=True) 

def resize_image(image, scale):
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #pil_image = Image.fromarray(resized_image)
    return resized_image

def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    num_faces = 0 if boxes is None else len(boxes)
    return num_faces

def process_image(image_path, scale):
    parts = os.path.basename(image_path).split('-')
    year_str = parts[1]
    try:
        year = int(year_str)
        decade = (year // 10) * 10
    except ValueError:
        return None, 0
    
    image = cv2.imread(image_path)
    resized_image = resize_image(image, scale)
    num_faces = detect_faces(resized_image)
    
    return decade, num_faces

def process_folder(folder_path, scale):
    data = {
        'Decade': [],
        'Total Pages': [],
        'Pages with Faces': [],
        '% Pages with Faces': [],
        'Total Faces': []
    }

    for root, dirs, files in os.walk(folder_path):
        for filename in tqdm(files, position=0, leave=True):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                #print(f'processing {filename}')
                image_path = os.path.join(root, filename)
                decade, num_faces = process_image(image_path, scale)
                #print(f'Decade: {decade}\nFaces found: {num_faces}\n')
                
                if decade is not None:
                    if decade not in data['Decade']:
                        data['Decade'].append(decade)
                        data['Total Pages'].append(0)
                        data['Pages with Faces'].append(0)
                        data['Total Faces'].append(0)
                        data['% Pages with Faces'].append(0)  # Initialize percentage column

                    index = data['Decade'].index(decade)
                    data['Total Pages'][index] += 1
                    data['Total Faces'][index] += num_faces
                    if num_faces > 0:
                        data['Pages with Faces'][index] += 1

        # Calculate the percentage of pages with faces
    for i in range(len(data['Decade'])):
        if data['Total Pages'][i] > 0:
            data['% Pages with Faces'][i] = (data['Pages with Faces'][i] / data['Total Pages'][i]) * 100

    df = pd.DataFrame(data, index=False)
    #sort the dataframe by decades, ascending
    df = df.sort_values(by='Decade', ascending=True)
    return df

def main():
    
    parser = argparse.ArgumentParser(description='Process images in a folder and compute statistics per decade.')
    parser.add_argument('-s', '--scale', required=False, type=float, default=1, help='Scale factor to resize the images (default: 0.5)')
    
    args = parser.parse_args()
    scale = args.scale

    start_time = time.time()  # Start the timer
    
    for folder in os.listdir(data_path):
        current_folder_path = os.path.join(data_path, folder)
        
        start_time = time.time()  # Start the timer
        
        print(f'Processing "{folder}" folder')
        df = process_folder(current_folder_path, scale)
    
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f'Time elapsed: {elapsed_time} seconds')

        df.to_csv(f'../out/{folder} stats.csv')

if __name__ == '__main__':
    main()
