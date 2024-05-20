import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the directory of the script
os.chdir(script_directory)

def line_chart(df, x, y, title):
    """
    Generates a line chart from a DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - x: Column name to be used as the x-axis.
    - y: Column name to be used as the y-axis.

    Returns:
    - plt: Matplotlib plot object.
    """
    # Extract data for plotting
    x_data = df[x]
    y_data = df[y]

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='r', label=y)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.xticks(x_data)

    plt.tight_layout(pad=0.5)

    # Annotate each point with its y-value
    for i, (x_val, y_val) in enumerate(zip(x_data, y_data)):
        plt.annotate(f'%{round(y_val,2)}', (x_val, y_val), textcoords="offset points", xytext=(0,10), ha='center')

    return plt

def process_csv_files_in_folder(folder_path):
    y_axis =  '% Pages with Faces'
    x_axis = 'Decade'
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            newspaper = filename.split()[0]
            csv_path = os.path.join(folder_path, filename)
            df = pd.read_csv(csv_path)
            #folder path is where the plot is saved
            plt = line_chart(df, x_axis, y_axis, f'% of Pages per Decade Containing Faces for {newspaper} Newspaper')
            plt.savefig(os.path.join(folder_path,f'{newspaper} plot3.png'))
            plt.close()

def main():
    folder_path = os.path.join('..','out')

    process_csv_files_in_folder(folder_path)
    print(f"Plots have been saved in the '{folder_path}' folder.")

if __name__ == "__main__":
    main()
