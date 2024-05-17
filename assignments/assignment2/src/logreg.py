import os
import cv2
import numpy as np
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from tensorflow.keras.datasets import cifar10

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

def preprocess_images(unprocessed_images):
    """
    Preprocess images by converting to grayscale, normalizing, and reshaping.
    """
    processed_images = []
    for image in tqdm(unprocessed_images):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX)
        processed_images.append(image)
    return np.array(processed_images).reshape(-1, 1024)

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test

def convert_labels(input_labels):
    """
    Convert numeric labels to object names.
    """
    label_dict = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', 
                  '4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', 
                  '9': 'truck'}
    return np.array([label_dict[str(label[0])] for label in input_labels])

def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression(max_iter=1000, random_state = 42, verbose=1)
    return logreg.fit(X_train, y_train)


def save_classification_report(report, filename):
    with open(f'../out/{filename}', 'w') as f:
        f.write(report)

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Preprocess images
    X_train_processed = preprocess_images(X_train)
    X_test_processed = preprocess_images(X_test)
    
    # Convert labels
    y_train_labels = convert_labels(y_train)
    y_test_labels = convert_labels(y_test)

    # Train logistic regression classifier and generate report
    classifier = train_logistic_regression(X_train_processed, y_train_labels)
    
    y_pred = classifier.predict(X_test_processed)

    report = classification_report(y_test_labels, y_pred)

    # Save classification report
    save_classification_report(report, 'logistic_regression_classification_report.txt')

if __name__ == "__main__":
    main()
