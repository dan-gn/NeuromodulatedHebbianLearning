import os
import pandas as pd

def append_line_to_csv(file_path, new_line_dict):
    if os.path.exists(file_path):
        # Read the existing CSV file
        df = pd.read_csv(file_path)
        # Append the new line (as a dictionary)
        df = pd.concat([df, pd.DataFrame([new_line_dict])])
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame([new_line_dict])

    # Save it to CSV
    df.to_csv(file_path, index=False)


if __name__ == "__main__":

    # Example usage:
    file_path = 'your_file.csv'
    new_line = {
        'column1': 'value1',
        'column2': 'value2',
        'column3': 'value3',
        # Add all columns and values you need
    }

    append_line_to_csv(file_path, new_line)
