import os
import argparse
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

data_folder = os.path.join('..', 'in', 'jpg')
output_folder = os.path.join('..', 'out')

# Adding argparse
parser = argparse.ArgumentParser(description='Process images in a folder and find similar images to a target image, using nearest neighbor.')
parser.add_argument('-t', '--target', required=True, type=int, help='Number of your desired target image.')
parser.add_argument('-s', '--similar', required=False, type=int, default=5, help='*OPTIONAL: How many similar images to find. Default=5.')

args = parser.parse_args()
target_image = args.target

def find_similar_images(target_image_path, data_folder, similar_images=5):
    """Find the most similar images to the target image in the dataset."""
    target_hist = calculate_histogram(target_image_path)

    results = []
    for file in tqdm(os.listdir(data_folder), colour='green'):
        if file.endswith('.jpg'):
            image_path = os.path.join(data_folder, file)
            hist = calculate_histogram(image_path)
            distance = compare_histograms(target_hist, hist)
            results.append((file, round(distance, 2)))

    # Sort the results based on the distance
    results.sort(key=lambda results: results[1])
    # Keep the top similar_images + 1 (including the target image itself)
    results = results[:similar_images + 1]
    for result in results:
        print(f'{result[0]}: Distance: {result[1]}')
    return results

def calculate_histogram(image_path):
    """Load an image & Calculate its color histogram."""
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    normalized_hist = cv2.normalize(hist, None, 0, 1.0, cv2.NORM_MINMAX)
    return normalized_hist

def compare_histograms(hist1, hist2):
    """Compare two histograms using the chi-square distance."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def save_results(results, output_folder):
    """Save the results to a CSV file."""
    df = pd.DataFrame(results, columns=['Filename', 'Distance'])
    df.to_csv(os.path.join(output_folder, 'similar_images_simple.csv'), index=False)

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
        ax.set_title(f"# {i}. {image_name}\nDistance: {distance}")
        ax.axis('off')
    
    # Hide any extra empty subplots
    for j in range(i + 1, num_rows * 3):
        row = j // 3
        col = j % 3
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'similar_simple.png'))
    plt.close()

def main():
    print(f'Finding similar images for image_{target_image:04d} using simple image search.')
    target_image_path = os.path.join(data_folder, f'image_{target_image:04d}.jpg')    #using :04d to pad the number with 0's to ensure it is 4 digits long.

    # Find similar images
    results = find_similar_images(target_image_path, data_folder, similar_images=args.similar)

    # Plot images
    plot_images(results)

    # Save results
    save_results(results, output_folder)

if __name__ == "__main__":
    main()
