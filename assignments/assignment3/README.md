# Assignment #3: CNN Transfer Learning With VGG16

## What is this?
This assignment features a script that uses transfer learning to train a VGG16 CNN (Convolutional Neural Network) to classify images of different document types, using the `Tobacco3482` dataset. The script also uses data augmentation to augment validation data "on the fly". This results in a classification report and a plot of the the loss & accuracy curves from the training process.

## Setup
1. Make sure to have python and Git Bash installed!

2. Open a Git Bash terminal and use Git to download the repository:

```sh
git clone https://github.com/missingusername/cds-vis-git.git
```

3. Navigate to the project folder for this assignment:

```sh
cd assignments/assignment3
```

4. Before doing anything else, you first need to get the tobacco dataset. You can download the `Tobacco3482` dataset manually from [Kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg). When downloaded, unzip the `archive.zip` and place the `Tobacco3482-jpg` folder inside the `in` folder of the `assignment3` directoy. However, remember to delete the duplicate `Tobacco3482-jpg` folder inside `Tobacco3482-jpg` itself. This should leave you with a file structure like this:

```
assignment3/
    ├── in/
    │   └── tobacco3482-jpg/
    │       ├── ADVE
    │       ├── Email
    │       ├── Form
    │       ├── Letter
    │       ├── Memo
    │       ├── News
    │       ├── Note
    │       ├── Report
    │       ├── Resume
    │       └── Scientific
    ├── out/
    │   ├── classification_report_with_vgg16.txt
    │   └── learning_curves_with_vgg16.png
    └── src/
        └── main.py
```

5. Before you can run the scripts, make sure you have the required libraries in the `requirements.txt`. This can be done by simply running the OS-appropriate setup script from inside the `assignment3` folder, which will set up a virtual environment and get the required libraries. Again, using Git Bash:

For Unix:
```sh
bash unix_setup.sh
```
For Windows:
```sh
bash win_setup.sh
```

6. The script can also take these optional command line arguments on execution:

| name        | flag | description                                                         | type | default value | Is required? |
|-------------|------|---------------------------------------------------------------------|------|---------------|--------------|
| save        | -s   | Save the model in the `/out` folder after training.                                      | bool | False         | OPTIONAL     |
| optimizer   | -o   | Which optimizer to use for the model.                               | str  | 'adam'        | OPTIONAL     |
| model       | -m   | Whether to print a summary of the model architecture.               | bool | True          | OPTIONAL     |
| epochs      | -e   | How many epochs to train the model over.                            | int  | 10            | OPTIONAL     |
| randomstate | -r   | What random state/"seed" to use when train-test splitting the data. |  int | 42            | OPTIONAL     |

7. To finally execute the script, simply run the OS-appropriate `run.sh` script in the same Git Bash terminal:

Unix: 
```sh
bash unix_run.sh
```
Windows: 
```sh
bash win_run.sh
```

## Takeaways from output

|       Class       | Precision | Recall | F1-Score | Support |
|:-----------------:|:---------:|:------:|:--------:|:-------:|
|        ADVE       |    0.96   |  0.84  |   0.90   |    57   |
|       Email       |    1.00   |  0.85  |   0.92   |   135   |
|        Form       |    0.73   |  0.67  |   0.70   |    88   |
|      Letter       |    0.73   |  0.69  |   0.71   |   122   |
|        Memo       |    0.51   |  0.83  |   0.63   |   109   |
|        News       |    0.75   |  0.97  |   0.85   |    34   |
|        Note       |    0.92   |  0.67  |   0.77   |    36   |
|      Report       |    0.59   |  0.56  |   0.57   |    48   |
|      Resume       |    0.62   |  0.53  |   0.57   |    15   |
|  Scientific       |    0.64   |  0.34  |   0.44   |    53   |
| **Accuracy**      |           |        |   0.73   |   697   |
| **Macro Avg**     |    0.74   |  0.70  |   0.71   |   697   |
| **Weighted Avg**  |    0.76   |  0.73  |   0.73   |   697   |

As we can see from the classification report, the final model appears decent at classifying different document categories, with a weighted average F1 score of **0.73**. This means that, on average, the model is good at classifying the different types of documents.
Looking at the specific classes, we can see that it seems especially good atclassifying **ADVE** and **Emails**, while seemingly less good at **Scientific**.

Looking at the loss curves of the models training process however, we can gain a better insight into how the model arrived at this performance, and whether or not there appears to be more "power" left to train out of it.

![Learning curves](out/learning%20curves.png)

### Loss curve
In the first plot, we see the loss curve. The loss curve represents the error associated with the model's predictions, ie. the loss function measures how well the model's predictions match the actual labels. Here, *lower* values indicate better performance, as there is less error associated with the predictions.

Looking at the training and validation curves, we see that both start out seemingly close to each other, and track each other fairly well over the training period. Here, we see a steep initial drop-off for both lines, progressively tapering off over the epochs.

A decrease in training loss indicates that the model is learning and fitting the training data better as training progresses.

A decrease in validation loss means that the model is also getting better at predicting the labels of data it has not encountered before.

With both curves following each other, it means that as the model is getting better at predicting on the training data, it is also getting better at predicting on new data.

### Accuracy curve
In the second plot, se see the accuracy curve. Accuracy represents the proportion of correct predictions out of the total predictions. So unlike loss, *higher* accuracy indicates better model performance, since a larger proportion of the predictions are correct.

Again the training and validation lines match start out close and follow each other fairly closely during training. Here, we see a steep initial increase in accuracy, again gradually tapering off as the training goes on over the epochs. This is to be expected, and reflects the decreasing loss curve.

The increase in training accuracy means that the model is getting better at predicting the labels of the training data. This is to be expected, since the model learns over time what the actual labels are for the data.

The increase in validation accuracy is important, since it refelects how well the model can generalize what its learned to new data, which it hasnt seen before.

Therefore, it is a good sign that the validation accuracy follows closely along with the training accuracy, since that means that the model generalizes well to new data.

## Limitations and possible steps to improvement
As in assignment 2, there is still the problem of finding a "perfect" model architecture, and even took a decent amount of time to arrive at this one. However, as mentioned in the improvements for A2, one could again have used a grid search to find better hyperparameters for the model.

Something else one could improve was changing the data augmentation parameters. I tried fiddling around with a few different things, such as zooming in & out, and shifting the image vertically and horizontally, however those appeared to *decrease* the final F1 score (*albeit very slightly*), so i opted to not use them. As with the model architecture, this is something that takes a great amount of time to fine tune, since you have to run the training process again each time, but unlike model architecture, there does not appear to be a grid-search like method of finding optimal data augmentation.