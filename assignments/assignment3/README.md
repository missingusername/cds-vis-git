# CDS Visual Analytics Assignment #3: Simple Image Search Algorithm (+KNN)

## What is this?


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
