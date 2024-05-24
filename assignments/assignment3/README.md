# CDS Visual Analytics Assignment #3: 

## What is this?

it uses data augmentation to

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

6. To finally execute the script, simply run the OS-appropriate `run.sh` script in the same Git Bash terminal:

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
|        ADVE       |    0.92   |  0.79  |   0.85   |    57   |
|       Email       |    0.85   |  0.84  |   0.84   |   135   |
|        Form       |    0.42   |  0.65  |   0.51   |    88   |
|      Letter       |    0.45   |  0.84  |   0.59   |   122   |
|        Memo       |    0.39   |  0.33  |   0.36   |   109   |
|        News       |    0.75   |  0.62  |   0.68   |    34   |
|        Note       |    0.58   |  0.19  |   0.29   |    36   |
|      Report       |    1.00   |  0.06  |   0.12   |    48   |
|      Resume       |    0.00   |  0.00  |   0.00   |    15   |
|  Scientific       |    0.47   |  0.13  |   0.21   |    53   |
| **Accuracy**      |           |        |   0.56   |   697   |
| **Macro Avg**     |    0.58   |  0.45  |   0.44   |   697   |
| **Weighted Avg**  |    0.60   |  0.56  |   0.53   |   697   |
From the f-1 scores we can see that the model actually does learn to distinguish fairly well between the differnt types of documents. 

![Learning curves](out/learning%20curves.png)
