# CDS Visual Analytics: Assignment #1

## Setup
1. Make sure to have python and Git Bash installed!

2. Open a Git Bash terminal and use Git to download the repository:

```sh
git clone https://github.com/missingusername/cds-vis-git.git
```

3. Navigate to the project folder for this assignment:

```sh
cd assignments
cd assignment1
```

4. Before running the program make sure you have opencv, Pandas, and TQDM installed. This can be done by simply running the ```setup.sh``` script from inside the ```assignment1``` folder, again using Git Bash:

```sh
./setup.sh
```

1. Before we can run the script, we first need to get the actual data. The data corpus can be downloaded by clicking [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz). When downloaded, unzip the folder and place the ```jpg``` folder in the ```in``` folder of the ```assignment1``` directoy.

2. To finally execute the file, simply run the script in the same Git Bash terminal:

```sh
python src/assignment1.py
```

## The Code
The script starts off by prompting you for a 4 digit code for the target image you want to find similar images for. Simply find an image in the ```jpg``` you want to search for. For example, ```image_0249.jpg``` would just require you to input ```0249``` in the terminal.

The script works by simply looping through every ```.jpg``` file in the ```jpg``` folder. For each file, we load it as an openCV image, create an RGB histogram for it, and then normalize the histogram afterwards.

When the normalized histogram has been made, we compare each one with the target images histogram using the ```HISTCOMP_CHISQR``` method. After comparing, we add the images name and its distance (rounded to 2 decimal points) to a tuple, and append that tuple to a results list.
When every image and its distance has been calculated and added to the list, we then sort the results and keep the top 6 (the target + the 5 most similar).

Then, we show the images using matplotlib, so we can see if they are actually close to the target image.
Lastly, we convert the results to a pandas dataframe, and save that as a ```.csv``` file in the ```out``` folder.
