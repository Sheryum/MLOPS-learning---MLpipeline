import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging # Set up logging
import dvc.api




#------------------------------------------------------ LOGGER CONFIGURATION ------------------------------------------------------##This code is used to configure the logger for the data ingestion module
#It will create 
#Ensure the directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#LOGGING CONFIGURATION
#To use a logger by creating a logger object
#This object will have methods including handlers and formatters
#A handler is responsible for sending the log messages to their final destination
#A formatter is responsible for formatting the log messages
#The handler can be divided into different types, such as console(stream), file, etc.
#In case of console, the log messages will be printed to the console
#nd in case of file, the log messages will be written to a file

#There are multiple log levels, such as DEBUG, INFO, WARNING, ERROR, CRITICAL
#DEBUG is the lowest level, and CRITICAL is the highest level

#Handler and a formatter are used to configure the logger

logger = logging.getLogger('data_ingestion') #This will create a logger named 'data_ingestion'
logger.setLevel(logging.DEBUG) #Set the logging level to INFO


#defining the console handler
console_handler = logging.StreamHandler() #This will create a console handler
console_handler.setLevel(logging.DEBUG) #Set the logging level for the console handler to INFO


#Definging the file handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log') #Define the path for the log file
file_handler = logging.FileHandler(log_file_path) #This will create a file handler
file_handler.setLevel(logging.DEBUG) #Set the logging level for the file handler to DEBUG

#Define the format of the log messages, below format will include the timestamp, logger name, log level, and the actual log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #This will format the log messages

console_handler.setFormatter(formatter) #Set the formatter for the console handler
file_handler.setFormatter(formatter) #Set the formatter for the file handler

#Add the handlers to the logger
logger.addHandler(console_handler) #Add the console handler to the logger
logger.addHandler(file_handler) #Add the file handler to the logger




#------------------------------------------------------ DATA INGESTION FUNCTION ------------------------------------------------------##This code is used to ingest the data from the source and save it to the destination

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a given URL and return it as a pandas DataFrame.
    
    Parameters:
    - data_url (str): The URL of the dataset to be loaded.
    
    Returns:
    - pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        logger.debug(f"Loading data from {data_url}")
        data = pd.read_csv(data_url, encoding='latin-1')
        logger.debug("Data loaded successfully")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parser error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected Error occured while loading data: {e}")
        raise


#------------------------------------------------------ INITIAL DATA PREPROCESSING FUNCTION ------------------------------------------------------##This code is used to do initial preprocessing of the data 
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.
    
    Parameters:
    - data (pd.DataFrame): The dataset to be preprocessed.
    
    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """
    try:
        logger.debug("Starting data preprocessing")
        # Handle missing values
        data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True) # Drop unnecessary columns
        data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True) # Rename columns for clarity
        logger.debug("Preprocessing Completed: Dropped unnecessary columns and renamed columns")
        return data
    except KeyError as e:
        logger.error(f"Missing Columns in Data frame: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise   
    
    
#------------------------------------------------------ SAVING PROCESSED DATA FUNCTION ------------------------------------------------------##This code is used to save the processed data to a file

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets to the specified path.
    
    Parameters:
    - train_data (pd.DataFrame): The training dataset.
    - test_data (pd.DataFrame): The testing dataset.
    - data_path (str): The path where the datasets will be saved.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw') # Create a directory for raw data
        os.makedirs(raw_data_path, exist_ok=True) # Ensure the directory exists
        train_file = os.path.join(raw_data_path, 'train.csv') # Create a file path for the training data
        test_file = os.path.join(raw_data_path, 'test.csv') # Create a file path for the testing data
        
        train_data.to_csv(train_file, index=False) # Save the training data to a CSV file
        test_data.to_csv(test_file, index=False) # Save the testing data to a CSV
        
        logger.debug(f"Train data saved to {train_file}")
        logger.debug(f"Test data saved to {test_file}")
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise
    
    
    
#------------------------------------------------------ MAIN FUNCTION ------------------------------------------------------##This code is used to run the data ingestion module

def main():
    """
    Main function to execute the data ingestion process.
    """
    try:
        params = dvc.api.params_show() # Load parameters from params.yaml
        test_size = params["data_ingestion"]["test_size"] # Define the test size for splitting the data
        random_state = params["data_ingestion"]["random_state"] # Define the random state for reproducibility
        data_url = 'https://github.com/Sheryum/DATASETS/raw/refs/heads/main/spam.csv' # URL of the dataset
        data_path = './data' # Path where the data will be saved
        # Load the data
        data = load_data(data_url)
        
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        # Split the data into train and test sets
        train_data, test_data = train_test_split(processed_data, test_size=test_size, random_state=random_state)
        
        # Save the processed data
        save_data(train_data, test_data, data_path)
        
        logger.info("Data ingestion process completed successfully")
    except Exception as e:
        logger.error(f"Data ingestion process failed: {e}")
        print(f"An error occurred: {e}")
        
        
if __name__ == "__main__":
    main() # Run the main function
#This will execute the data ingestion process when the script is run directly