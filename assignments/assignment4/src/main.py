import os
import pandas as pd

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

data_path = os.path.join('..','in','images','images')
gdl_path = os.path.join(data_path, 'GDL')

def count_pages_per_decade(folder_path):
    # Initialize a dictionary to store counts per decade
    decade_counts = {}

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):  # Add other image formats if needed
            # Split the filename
            parts = filename.split('-')
            
            # Extract the year
            year_str = parts[1]
            try:
                year = int(year_str)
                decade = (year // 10) * 10  # Calculate the decade
            except ValueError:
                # Skip files with an invalid year part
                continue

            # Update the counts for the decade
            if decade in decade_counts:
                decade_counts[decade] += 1
            else:
                decade_counts[decade] = 1

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(decade_counts.items()), columns=['Decade', 'Page Count'])

    return df

# Example usage
df = count_pages_per_decade(gdl_path)
print(df)
