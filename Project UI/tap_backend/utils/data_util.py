import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        # Attempt to load the data from the CSV file
        data = pd.read_csv(file_path, header=0, index_col=0)
        return data
    except FileNotFoundError:
        # If the file is not found, print an error message and return None
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        # If the file is empty, print an error message and return None
        print(f"Error: The file {file_path} is empty.")
        return None
    except pd.errors.ParserError:
        # If there is an error parsing the file, print an error message and return None
        print(f"Error: The file {file_path} could not be parsed.")
        return None
    except Exception as e:
        # Catch any other exceptions and print an error message
        print(f"An unexpected error occurred: {e}")
        return None

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing rows with missing values.

    Args:
        data (pd.DataFrame): The data to be cleaned.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    # Remove rows with missing values
    date_to_id = {date: idx + 1 for idx, date in enumerate(data['DATE'].unique())}

    data['ACCNUM'] = data.groupby('DATE')['ACCNUM'].transform(
        lambda group: group.fillna(date_to_id[group.name])
    ).reset_index(drop=True)
    
    data['MONTH'] = pd.to_datetime(data['DATE']).dt.month
    data['YEAR'] = pd.to_datetime(data['DATE']).dt.year
    data['DAY'] = pd.to_datetime(data['DATE']).dt.day

    boolean_columns = [
        'PEDESTRIAN',
        'CYCLIST',
        'AUTOMOBILE',
        'MOTORCYCLE',
        'TRUCK',
        'TRSN_CITY_VEH',
        'EMERG_VEH',
        'PASSENGER',
        'SPEEDING',
        'AG_DRIV',
        'REDLIGHT',
        'ALCOHOL',
        'DISABILITY'
    ]
    
    data[boolean_columns] = data[boolean_columns].fillna("No")
    data['ROAD_CLASS'] = data['ROAD_CLASS'].str.replace(r'MAJOR ARTERIAL ', 'MAJOR ARTERIAL', regex=False)

    categorical_columns = data.select_dtypes(include=[object, 'category']).columns.tolist()
    data[categorical_columns] = data[categorical_columns].apply(lambda col: col.str.upper())


    return data