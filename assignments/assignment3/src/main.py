import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

#disabling warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# image processsing
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# VGG16 model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

# Set the path to the input and output directories
in_path = os.path.join('..', 'in')
data_path = os.path.join(in_path, 'Tobacco3482-jpg')
output_path = os.path.join('..', 'out')

# Define the filename for the saved data
saved_data_file = os.path.join(output_path, 'saved_data.pkl')

parser = argparse.ArgumentParser(description='Use transfor learning on a VGG16 model, to teach it to distinguish between certain document types.')
parser.add_argument('-s', '--save', required=False, type=bool, default=False, help='*OPTIONAL: Save the model after training. Default = False')
parser.add_argument('-o', '--optimizer', required=False, type=str.lower, default='adam', help='*OPTIONAL: Which optimizer to use (Only takes SGD or Adam). Default = Adam')
parser.add_argument('-m', '--model', required=False, type=bool, default=True, help='*OPTIONAL: Get a summary of the model. Default = True')
parser.add_argument('-e', '--epochs', required=False, type=int, default=10, help='*OPTIONAL: How many epochs to train the model over. Default = 10.')
parser.add_argument('-r', '--randomstate', required=False, type=int, default=42, help='*OPTIONAL: What random state/"seed" to use when train-test splitting the data. Default = 42.')

args = parser.parse_args()

# Function to load a pe-made data file if it exists, or generate one if it doesnt.
def load_or_create_data(data_path, saved_data_file):
    if os.path.exists(saved_data_file):
        print("Loading saved data...")
        with open(saved_data_file, 'rb') as f:
            data, labels = pickle.load(f)
        print('Data loaded!')
    else:
        print("Generating data file from images...")
        data, labels = generate_data(data_path)
        print('Saving data...')
        with open(saved_data_file, 'wb') as f:
            pickle.dump((data, labels), f)
        print('Data saved in /out folder!')
    return data, labels

# Function to load data from images
def generate_data(data_path):
    data = []
    labels = []

    for folder in tqdm(sorted(os.listdir(data_path)), desc='Processing folders...'):
        folder_path = os.path.join(data_path, folder)
        for image_file in tqdm(sorted(os.listdir(folder_path)), desc=f'Processing folder: {folder}'):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(folder_path, image_file)
                image = load_img(image_path, target_size=(224, 224))  
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)
                labels.append(folder)

    return np.array(data), np.array(labels)

def build_model():
    # load vgg16 model without classifier layers
    base_model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in base_model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(base_model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(128, activation='relu')(bn)
    drop = Dropout(0.1)(class1)
    output = Dense(10, activation='softmax')(drop)

    # define new model
    new_model = Model(inputs=base_model.inputs, outputs=output)

    # Initializes the learning_rate_schedule 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, 
                                                                    decay_steps=10000,
                                                                    decay_rate=0.9)
    
    if args.optimizer == 'adam':
        # Initializes the adam optimizer 
        model_optimizer = Adam(learning_rate=lr_schedule)
    if args.optimizer == 'sgd':
        # Initializes the adam optimizer 
        model_optimizer = SGD(learning_rate=lr_schedule)

    # compile new model 
    new_model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # optionally print a summary of the new models architecture
    if args.model:
        new_model.summary()

    return new_model

def setup_datagenerator():
    datagen = ImageDataGenerator(horizontal_flip=True, #horizontally flip images
                                rotation_range=20,  #rotate images up to 20 degrees
                                fill_mode='nearest', #fill empty space with the nearest border color of the image
                                brightness_range=[0.9,1.1], #change image brightness between 90%-110%
                                validation_split=0.1) #using 10% of the data as a validation split, meaning we test the model against the augmented images
    return datagen

def train_model(model, X_train, y_train, datagen):   

    # Fits the model on the training and validation data using "datagen" for data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=128), 
                                validation_data = datagen.flow(X_train, y_train, 
                                                                batch_size=128, 
                                                                subset = "validation"),
                                                                epochs=args.epochs,
                                                                verbose=1) 

    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, lb):
    predictions = model.predict(X_test, batch_size=128)
    report = classification_report(y_test.argmax(axis=1), 
                                    predictions.argmax(axis=1), 
                                    target_names=lb.classes_)
    print(report)
    with open(os.path.join(output_path, 'classification_report_with_vgg16.txt'), 'w') as f:
        f.write(report)

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
    plt.savefig(os.path.join(output_path,'learning curves.png'))

# Main function
def main():
    # Load data & labels, or create a new .pkl data file of the chosen input data  
    data, labels = load_or_create_data(data_path, saved_data_file)
    # Encoding labels before train-test splitting
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=args.randomstate)
    # Build model
    model = build_model()
    # setting up a data augmentation
    datagen = setup_datagenerator()
    #training the model and getting the training history
    model, history = train_model(model, X_train, y_train, datagen)
    # Plot learning curves
    plot_history(history, args.epochs)
    # Evaluate model
    evaluate_model(model, X_test, y_test, lb)

    #optionally save the model
    if args.save:
        model.save(os.path.join(output_path, 'tobacco_vgg16.h5'))

if __name__ == '__main__':
    main()
