import os
import logging # Set up logging
import pandas as pd
import numpy as np
import pickle
import json
from dvclive import Live
from dvc.api import params_show

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

#------------------------------------------------------ LOGGER CONFIGURATION ------------------------------------------------------##This code is used to configure the logger for the data ingestion module
#It will create 
#Ensure the directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation') #This will create a logger named 'model_evaluation'
logger.setLevel(logging.DEBUG) #Set the logging level to INFO


#defining the console handler
console_handler = logging.StreamHandler() #This will create a console handler
console_handler.setLevel(logging.DEBUG) #Set the logging level for the console handler to INFO


#Definging the file handler
log_file_path = os.path.join(log_dir, 'model_evaluation.log') #Define the path for the log file
file_handler = logging.FileHandler(log_file_path) #This will create a file handler
file_handler.setLevel(logging.DEBUG) #Set the logging level for the file handler to DEBUG

#Define the format of the log messages, below format will include the timestamp, logger name, log level, and the actual log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #This will format the log messages

console_handler.setFormatter(formatter) #Set the formatter for the console handler
file_handler.setFormatter(formatter) #Set the formatter for the file handler

#Add the handlers to the logger
logger.addHandler(console_handler) #Add the console handler to the logger
logger.addHandler(file_handler) #Add the file handler to the logger



#-------------------------------------------------------- MODEL EVALUATION FUNCTIONS ------------------------------------------------------##This code is used to evaluate the performance of a trained machine learning model using various metrics



def load_model(model_path: str):
    """
    Load a trained machine learning model from a file.
    
    Parameters:
    - model_path (str): The path to the model file.
    
    Returns:
    - model: The loaded machine learning model.
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model
    
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

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
        logger.error(f"Failed to parse CSV file: {e}")
        raise
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise   
    
    except Exception as e:
        logger.error(f"Unexpected error during loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the performance of a trained machine learning model using various metrics.
    
    Parameters:
    - model: The trained machine learning model.
    - X_test (np.ndarray): The test features.
    - y_test (np.ndarray): The test labels.
    
    Returns:
    - dict: A dictionary containing the evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)  # Make predictions on the test set
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        
        logger.debug(f"Model evaluation metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    

def save_evaluation_results(metrics: dict, output_path: str)-> None:
    """
    Save the evaluation metrics to a JSON file.
    
    Parameters:
    - metrics (dict): The evaluation metrics to save.
    - output_path (str): The path to save the JSON file.
    """
    try:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        with open(output_path, 'w') as file:
            json.dump(metrics, file)
        logger.debug(f"Evaluation results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise
    
    
    
def main():
    """
    Main function to run the model evaluation module.
    It will load the model, load the test data, evaluate the model, and save the evaluation results.
    """
    try:
        
        params= params_show()  # Load parameters from DVC params file
        
        model_path = './models/model.pkl'  # Path to the trained model
        test_data_path = './data/processed/test_tfidf.csv'  # Path to the processed test data
        output_path = './reports/metrics.json'  # Path to save the evaluation results

        model = load_model(model_path)  # Load the trained model
        test_data = load_data(test_data_path)  # Load the test data
        
        X_test = test_data.drop(columns=['target']).values  # Extract features
        y_test = test_data['target'].values  # Extract labels
        
        metrics = evaluate_model(model, X_test, y_test)  # Evaluate the model
        
        
#        with Live(save_dvc_exp=True) as live:
#            live.log_metric('accuracy', metrics['accuracy'])
#            live.log_metric('precision', metrics['precision'])
#            live.log_metric('recall', metrics['recall'])
#            live.log_metric('roc_auc', metrics['roc_auc'])
#            
#            live.log_params(params)
        
        save_evaluation_results(metrics, output_path)  # Save the evaluation results
        
        
    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        raise
    
    
if __name__ == "__main__":
    main()  # Run the main function to execute the model evaluation module
