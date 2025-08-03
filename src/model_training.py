import os
import logging # Set up logging
import pandas as pd
import numpy as np
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier




def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its content as a dictionary.
    
    Parameters:
    - file_path (str): The path to the YAML file.
    
    Returns:
    - dict: The content of the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"YAML file loaded successfully from {file_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading YAML file: {e}")
        raise

#------------------------------------------------------ LOGGER CONFIGURATION ------------------------------------------------------##This code is used to configure the logger for the data ingestion module
#It will create 
#Ensure the directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training') #This will create a logger named 'model_training'
logger.setLevel(logging.DEBUG) #Set the logging level to INFO


#defining the console handler
console_handler = logging.StreamHandler() #This will create a console handler
console_handler.setLevel(logging.DEBUG) #Set the logging level for the console handler to INFO


#Definging the file handler
log_file_path = os.path.join(log_dir, 'model_training.log') #Define the path for the log file
file_handler = logging.FileHandler(log_file_path) #This will create a file handler
file_handler.setLevel(logging.DEBUG) #Set the logging level for the file handler to DEBUG

#Define the format of the log messages, below format will include the timestamp, logger name, log level, and the actual log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #This will format the log messages

console_handler.setFormatter(formatter) #Set the formatter for the console handler
file_handler.setFormatter(formatter) #Set the formatter for the file handler

#Add the handlers to the logger
logger.addHandler(console_handler) #Add the console handler to the logger
logger.addHandler(file_handler) #Add the file handler to the logger




#------------------------------------------------------ MODEL TRAINING FUNCTIONS ------------------------------------------------------##This code is used to train a machine learning model using the training data and save the model to a file



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    - file_path (str): The path to the CSV file.
    
    Returns:
    - pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}, {data.shape}")
        return data
    
    except pd.errors.ParserError as e:
        logger.error(f"Failed to Parse to CSV file: {e}")
        raise
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected Error during loading data from {file_path}: {e}")
        raise
    
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:  
    
    """
    Train a Random Forest model using the provided training data and parameters.
    
    Parameters:
    - X_train (np.ndarray): The training features.
    - y_train (np.ndarray): The training labels.
    - params (dict): The parameters for the Random Forest model.
    
    Returns:
    - RandomForestClassifier: The trained model.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
    
        logger.debug("Initializing Random Forest classifiier with parameters: %s" , params)
        
        clf = RandomForestClassifier(**params)  # Initialize the model with the provided parameters
        
        logger.debug("Fitting the model to the training data of size %s", X_train.shape[0])
        clf.fit(X_train, y_train)  # Fit the model to the training data
        logger.debug("Model training completed successfully")
        
        
        return clf  # Return the trained model
    
    except ValueError as e:
        logger.error(f"Value Error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise
    
    
    
def save_model(model, file_path: str) -> None:
    
    """
    Save the trained model to a file.
    
    Parameters:
    - model: The trained model to be saved.(using pickle)
    - file_path (str): The path where the model will be saved.
    """
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)  # Save the model using pickle
            
        logger.debug(f"Model saved successfully to {file_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Failed to save the model to {file_path}: {e}")
        raise
    
    
def main():
    """
    Main function to run the model training module.
    It will load the training data, train the model, and save the trained model.
    """
    try:
        train_data_path = './data/processed/train_tfidf.csv'  # Path to the preprocessed training data
        model_save_path = './models/model.pkl'  # Path to save the trained model
        
        # Load the training data
        train_data = load_data(train_data_path)
        
        # Extract features and target variable
        X_train = train_data.iloc[:,:-1].values  # Assuming 'target' is the label column
        y_train = train_data.iloc[:,-1].values
        
        logger.debug("Training data loaded successfully")
        
        # Define model parameters
        params = load_yaml("params.yaml")  # Load parameters from DVC
        params = params["model_training"]
        
        # Train the model
        model = train_model(X_train, y_train, params)
        
        # Save the trained model
        save_model(model, model_save_path)
        
    except Exception as e:
        logger.error(f"Failed to complete the model training process: {e}")
        raise
    

if __name__ == "__main__":
    main()  # Run the main function to execute the model training module