# base tools
import os
import pickle
import argparse

# data analysis
import pandas as pd
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

# tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

data_folder = os.path.join('..','in','jpg')
output_folder = os.path.join('..', 'out')
feature_list_path = os.path.join(output_folder, 'feature_list.pkl')

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#adding argparse
parser = argparse.ArgumentParser(description='Process images in a folder and find similar images to a target image, using nearest neighbor.')
parser.add_argument('-t', '--target', required=True, type=int, default=1, help='*REQUIRED: Number code of your desired target image. E.g., "image_1071.jpg" = -n 1071')
parser.add_argument('-s', '--similar', required=False, type=int, default=5, help='*OPTIONAL: How many similar images to find. Default=5.')


args = parser.parse_args()
target_image = args.target  # subtracting 1 from the inputted target code, since the filenames start at 0001, but indexing starts at 0, meaning there's an offset of 1.


# Globally defining what model to use for feature extraction
model = VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))

def extract_features(img_path, model):
    """
    Extract features from image data using pretrained model (e.g. VGG16)
    """
    input_shape = (224, 224, 3)
    img = load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img, verbose=False)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(features)
    return normalized_features

def extract_folder_features(input_folder):
    feature_list = []
    file_list = []

    for file_name in tqdm(os.listdir(input_folder), colour='green'):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(input_folder, file_name)
            file_list.append(file_path)
            feature_list.append(extract_features(file_path, model))
    
    return feature_list, file_list

def save_feature_list(feature_list, file_list, path):
    with open(path, 'wb') as f:
        pickle.dump((feature_list, file_list), f)

def load_feature_list(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_images(results):
    num_results = len(results)
    num_rows = (num_results + 2) // 3  # Calculate number of rows needed, rounding up

    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 8))
    
    for i, (image_name, distance) in enumerate(results):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        img = mpimg.imread(os.path.join(data_folder,image_name))
        ax.imshow(img)
        ax.set_title(f"# {i}. {image_name}\nDistance: {distance:.4f}")      #rounding distance to 4 decimals since KNN is much more accurate than the simple search
        ax.axis('off')

    # Hide any extra empty subplots
    for j in range(i + 1, num_rows * 3):
        row = j // 3
        col = j % 3
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'similar_knn.png'))
    plt.close()

def find_similar_images_knn(target_image_path, dataset_folder, similar_images=5):
    """
    Find the most similar images to the target image in the dataset using KNN.
    Returns a list of tuples (image_path, distance).
    """
    # Load or extract feature list
    if os.path.exists(feature_list_path):
        feature_list, file_list = load_feature_list(feature_list_path)
        print("Loaded feature list from file.")
    else:
        feature_list, file_list = extract_folder_features(dataset_folder)
        save_feature_list(feature_list, file_list, feature_list_path)
        print("Extracted and saved feature list.")

    # Fit the feature list to the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=similar_images + 1, 
                                 algorithm='brute',
                                 metric='cosine').fit(feature_list)

    # Calculate distances and indices for the target image
    target_features = extract_features(target_image_path, model)
    distances, indices = neighbors.kneighbors([target_features])

    results = []
    for i in range(similar_images+1):
        image_index = indices[0][i]
        image_name = os.path.basename(file_list[image_index])
        distance = distances[0][i]
        results.append((image_name, distance))

    return results

def save_results(results, output_folder):
    """Save the results to a CSV file."""
    df = pd.DataFrame(results, columns=['Filename', 'Distance'])
    df.to_csv(os.path.join(output_folder, 'similar_images_KNN.csv'), index=False)

def main():
    target_image_path = os.path.join(data_folder, f'image_{target_image:04d}.jpg')  # Adjust for 1-based indexing of images

    print(f'Finding similar images for image_{target_image:04d} using KNN')

    # Find similar images using KNN
    results = find_similar_images_knn(target_image_path, data_folder, similar_images=args.similar)

    # Save results to CSV
    save_results(results, output_folder)

    # Plot similar images
    plot_images(results)

if __name__ == "__main__":
    main()
