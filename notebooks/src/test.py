import os
import pandas as pd
import numpy as np
import argparse

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

def file_loader():
    parser = argparse.ArgumentParser(description = 'loading and printing an array')
    parser.add_argument('--input',
                        '-i',
                        required=True,
                        help = 'filepath to csv for loading ')

    args = parser.parse_args()

    return args

def main():
    args = file_loader()
    filepath = os.path.join('..',
                        '..',
                        '..',
                        '..',
                        'cds-vis-data',
                        'data',
                        'sample-data',
                        args.input)

    print(filepath)

    data = pd.read_csv(filepath)

    print(data)

if __name__ == '__main__':
    main()