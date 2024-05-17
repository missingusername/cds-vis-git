import os
import cv2
import numpy as np
from tqdm import tqdm
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

# Function to preprocess images
def preprocess_images(unprocessed_images):
    processed_images = []
    for image in tqdm(unprocessed_images):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX)
        processed_images.append(image)
    return np.array(processed_images).reshape(-1, 1024)

# Function to load CIFAR-10 dataset
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test

# Function to convert numeric labels to object names
def convert_labels(input_labels):
    label_dict = {'0': 'airplane', '1': 'automobile', '2': 'bird', '3': 'cat', 
                  '4': 'deer', '5': 'dog', '6': 'frog', '7': 'horse', '8': 'ship', 
                  '9': 'truck'}
    return np.array([label_dict[str(label[0])] for label in input_labels])

# Function to train MLP classifier
def train_mlp(X_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(128,), 
                        random_state=42, 
                        max_iter=1000, 
                        activation='logistic',
                        early_stopping=True,
                        verbose=True)
    mlp.fit(X_train, y_train)
    return mlp

# Function to save classification report
def save_classification_report(report, filename):
    with open(f'../out/{filename}', 'w') as f:
        f.write(report)

# Function to save loss curve plot during training
def save_loss_curve(losses, filename):
    plt.plot(losses)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig(f'../out/{filename}')
    plt.close()

# Main function
def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Preprocess images
    X_train_processed = preprocess_images(X_train)
    X_test_processed = preprocess_images(X_test)
    
    # Convert labels
    y_train_labels = convert_labels(y_train)
    y_test_labels = convert_labels(y_test)

    # Train MLP classifier
    mlp = train_mlp(X_train_processed, y_train_labels)

    # Save loss curve
    save_loss_curve(mlp.loss_curve_, 'mlp_loss_curve.png')

    # Predict and generate classification report
    y_pred = mlp.predict(X_test_processed)
    report = classification_report(y_test_labels, y_pred)

    # Save classification report
    save_classification_report(report, 'mlp_classification_report.txt')

if __name__ == "__main__":
    main()
