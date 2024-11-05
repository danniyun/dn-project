import os
import pickle
import pandas as pd


def save_mmep_data_to_file(data_dir, fields, dates, output_file):
    """
    Combine multiple CSV files into MMEP format and save the result as a pickle file.

    :param data_dir: Directory where the CSV files are stored
    :param fields: List of field names
    :param dates: List of dates
    :param output_file: Path to the output file where the combined data will be saved
    """
    all_data = []  # Store data for each day

    for date in dates:
        date_str = f'{date}'  # Format the date
        date_data = {}

        # Iterate over each field and load the data into the dictionary
        for field in fields:
            file_path = os.path.join(data_dir, f'{field}_{date_str}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df.apply(pd.to_numeric, errors='coerce')  # Ensure data is numeric
                date_data[field] = df
            else:
                continue

        if date_data:  # If data is successfully loaded
            combined_data = pd.concat(date_data.values(), axis=1, keys=date_data.keys())
            combined_data['didx'] = date  # Set the date as an index
            combined_data['tidx'] = combined_data.index  # Set minute index (row number)
            all_data.append(combined_data)

    # Ensure there is data to concatenate
    if all_data:
        # Combine data from all dates
        mmep_data = pd.concat(all_data, ignore_index=True)

        # Set MultiIndex as (didx, tidx)
        mmep_data.set_index(['didx', 'tidx'], inplace=True)

        # Remove columns containing 'Minutes', which apply to all fields
        mmep_data = mmep_data.drop(columns=[col for col in mmep_data.columns if 'Minutes' in col])

        # Save the combined data to a pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(mmep_data, f)

        print(f"Data saved to {output_file}")
    else:
        print("No data to concatenate. Please check the file paths or field names.")

def generate_dates_range(start_date, end_date):
    """
    Generate a list of dates from start_date to end_date in the format 'mmdd'.
    
    :param start_date: Start date, e.g., 401 for April 1st.
    :param end_date: End date, e.g., 1209 for December 9th.
    :return: A list of dates formatted as ['0401', '0402', ..., '1209'].
    """
    return [f"{month:02d}{day:02d}" for month in range(4, 13) for day in range(1, 32)
            if f"{month:02d}{day:02d}" >= f"{start_date:04d}" and f"{month:02d}{day:02d}" <= f"{end_date:04d}"]

def load_mmep_data_from_file(data_file):
    """
    Load MMEP data from a local file.
    
    :param data_file: The MMEP data file (in pickle format).
    :return: The loaded MMEP data.
    """
    with open(data_file, 'rb') as f:
        mmep_data = pickle.load(f)
    
    return mmep_data

def get_mmep_data(data_dir, fields, dates, output_file):
    """
    If the combined data file already exists locally, load it; otherwise, combine CSV files and save it.
    
    :param data_dir: Directory where the CSV files are stored.
    :param fields: List of field names (columns).
    :param dates: List of dates for which data is needed.
    :param output_file: Path to save the combined MMEP data file.
    :return: The loaded MMEP data.
    """
    if os.path.exists(output_file):
        print(f"Loading data from {output_file}")
        return load_mmep_data_from_file(output_file)
    else:
        print(f"Combining data from CSV files and saving to {output_file}")
        save_mmep_data_to_file(data_dir, fields, dates, output_file)
        return load_mmep_data_from_file(output_file)
