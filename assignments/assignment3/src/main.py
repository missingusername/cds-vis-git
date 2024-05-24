import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

# Set the path to the input and output directories
data_path = os.path.join('..', 'in', 'Tobacco3482-jpg')
output_path = os.path.join('..', 'out')

# Define the filename for the saved data
saved_data_file = os.path.join(output_path, 'saved_data.pkl')

# Function to load or save data
def load_or_save_data(data_path, saved_data_file):
    """
    Load data from images or load saved data if available.

    Args:
        data_path (str): Path to the directory containing image data.
        saved_data_file (str): Path to the saved data file.

    Returns:
        tuple: Tuple containing data and labels.
    """
    if os.path.exists(saved_data_file):
        print("Loading saved data...")
        with open(saved_data_file, 'rb') as f:
            data, labels = pickle.load(f)
        print('Data loaded!')
    else:
        print("Loading data from images...")
        data, labels = load_data(data_path)
        print('Saving data...')
        with open(saved_data_file, 'wb') as f:
            pickle.dump((data, labels), f)
        print('Data saved in /out folder!')
    return data, labels

# Function to load data from images
def load_data(data_path):
    """
    Load image data from the specified directory.

    Args:
        data_path (str): Path to the directory containing image data.

    Returns:
        tuple: Tuple containing data and labels.
    """
    data = []
    labels = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for image_file in tqdm(os.listdir(folder_path)):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                image = load_img(image_path, target_size=(224, 224))  
                image_array = img_to_array(image)
                data.append(image_array)
                labels.append(folder)

    return np.array(data), np.array(labels)

# Function to build the model
def build_model():
    """
    Build the VGG16-based model for image classification.

    Returns:
        Model: Compiled Keras model.
    """
    base_model = VGG16(include_top=False,
                       pooling='avg',
                       input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False
    flat1 = Flatten()(base_model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class2 = Dense(128, activation='relu')(bn)
    output = Dense(10, activation='softmax')(class2)
    model = Model(inputs=base_model.inputs, outputs=output)
    return model

# Function to train the model
def train_model(model, X_train, y_train, epochs=10):
    """
    Train the specified model on the training data.

    Args:
        model (Model): Keras model to train.
        X_train (ndarray): Training data.
        y_train (ndarray): Training labels.

    Returns:
        tuple: Tuple containing the trained model and training history.
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, 
                        validation_split=0.1, 
                        batch_size=128, 
                        epochs=epochs, 
                        verbose=1)
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, lb):
    """
    Evaluate the trained model on the test data.

    Args:
        model (Model): Trained Keras model.
        X_test (ndarray): Test data.
        y_test (ndarray): Test labels.
        lb (LabelBinarizer): LabelBinarizer object used for encoding.

    Returns:
        None
    """
    predictions = model.predict(X_test, batch_size=128)
    report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
    print(report)
    with open(os.path.join(output_path, 'classification_report_with_vgg16.txt'), 'w') as f:
        f.write(report)

# Function to plot learning curves
def plot_history(H, epochs):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()

# Main function
def main():
    # Load or save data
    X, y = load_or_save_data(data_path, saved_data_file)
    # Normalize pixel values
    X = X.astype('float') / 255.0
    # Encode labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build model
    model = build_model()
    # Train model
    epochs = 10
    model, history = train_model(model, X_train, y_train, epochs)
    # Save model
    model.save(os.path.join(output_path, 'tobacco_vgg16.h5'))
    # Evaluate model
    evaluate_model(model, X_test, y_test, lb)
    # Plot learning curves
    plot_history(history, epochs)

if __name__ == '__main__':
    main()
