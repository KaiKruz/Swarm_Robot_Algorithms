import os
import pandas as pd

def load_raw_data(directory="data/raw"):
    """Loads raw data files from the specified directory."""
    data = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            filepath = os.path.join(directory, file)
            data.append(pd.read_csv(filepath))
    if not data:  # Check if no files were added
        raise ValueError("No CSV files found in the directory to load.")
    return pd.concat(data, ignore_index=True)

def process_data(raw_data):
    """Processes raw data into a cleaned format."""
    processed_data = raw_data.dropna()
    return processed_data

if __name__ == "__main__":
    try:
        raw_data = load_raw_data()
        processed_data = process_data(raw_data)
        os.makedirs("data/processed", exist_ok=True)  # Ensure the processed directory exists
        processed_data.to_csv("data/processed/cleaned_data.csv", index=False)
        print("Data processing complete.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")