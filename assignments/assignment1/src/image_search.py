

#REMEMBER TO ADD A SECOND SCRIPT THAT UTILIZES NEAREST NEIGHBOR (NOTEBOOK 10)

import os
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# pip install opencv-python pandas tqdm

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

def find_similar_images(target_image_path, dataset_folder, similar_images=5):
    """Find the most similar images to the target image in the dataset."""
    target_hist = calculate_histogram(target_image_path)

    results = []
    for file in tqdm(os.listdir(dataset_folder), colour='green'):
        if file.endswith('.jpg'):
            image_path = os.path.join(dataset_folder, file)
            hist = calculate_histogram(image_path)
            distance = compare_histograms(target_hist, hist)
            results.append((file, round(distance, 2)))
    # sort the results list based on the distance (which is the second element in each tuple in the result list)
    results.sort(key=lambda results: results[1])
    # then keep the amount of similar images specified (by default 5) + 1 because we also keep the target.
    results = results[:similar_images+1]
    for result in results:
        print(f'{result[0]}: Distance: {result[1]}')
    return results

def calculate_histogram(image_path):
    """Load an image & Calculate its color histogram."""
    #loading the image with opencv
    image = cv2.imread(image_path)
    #creating a historgram of the images RGB channels
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    #normalizing the histogram
    normalized_hist = cv2.normalize(hist, None, 0, 1.0, cv2.NORM_MINMAX)
    return normalized_hist

def compare_histograms(hist1, hist2):
    """Compare two histograms using the chi-square distance."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def save_results(results, output_folder):
    """Save the results to a CSV file."""
    # Select the top 5 most similar images
    df = pd.DataFrame(results, columns=['Filename', 'Distance'])
    df.to_csv(os.path.join(output_folder, 'similar_images.csv'), index=False)

def plot_images(image_data):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, (image_name, distance) in enumerate(image_data):
        image_path = os.path.join('..',
                                  'in',
                                  'jpg',
                                  image_name)
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(f"{image_name}\nDistance: {distance}")
        ax.axis('off')
    
    plt.show()

def main():
    # Define paths and parameters
    dataset_folder = os.path.join('..',
                                  'in',
                                  'jpg')
    
    user_input = input('Enter the 4 digit code of your target image: ')
    target_image = f'image_{user_input}.jpg'
    print(f'Finding similar images for: {target_image}')

    target_image_path = os.path.join(dataset_folder, target_image)
    
    output_folder = os.path.join('..',
                                'out')

    # Find similar images
    results = find_similar_images(target_image_path, dataset_folder)

    #plot and show the images found
    plot_images(results)

    # Save results
    save_results(results, output_folder)

if __name__ == "__main__":
    main()
