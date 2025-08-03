import pandas as pd
import os
import logging # Set up logging


from sklearn.feature_extraction.text import TfidfVectorizer




#------------------------------------------------------ LOGGER CONFIGURATION ------------------------------------------------------##This code is used to configure the logger for the data ingestion module
#It will create 
#Ensure the directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_enigineering') #This will create a logger named 'feature_enigineering'
logger.setLevel(logging.DEBUG) #Set the logging level to INFO


#defining the console handler
console_handler = logging.StreamHandler() #This will create a console handler
console_handler.setLevel(logging.DEBUG) #Set the logging level for the console handler to INFO


#Definging the file handler
log_file_path = os.path.join(log_dir, 'feature_engineering.log') #Define the path for the log file
file_handler = logging.FileHandler(log_file_path) #This will create a file handler
file_handler.setLevel(logging.DEBUG) #Set the logging level for the file handler to DEBUG

#Define the format of the log messages, below format will include the timestamp, logger name, log level, and the actual log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #This will format the log messages

console_handler.setFormatter(formatter) #Set the formatter for the console handler
file_handler.setFormatter(formatter) #Set the formatter for the file handler

#Add the handlers to the logger
logger.addHandler(console_handler) #Add the console handler to the logger
logger.addHandler(file_handler) #Add the file handler to the logger




#------------------------------------------------------ FEATURE ENGINEERING FUNCTION ------------------------------------------------------##This code is used to perform feature engineering on the data, including encoding categorical variables and removing duplicates

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
        data.fillna('', inplace=True)  # Fill NaN values with empty strings
        logger.debug(f"Data loaded from {file_path}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise   
    
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF transformation to the specified text column in both training and testing data.
    
    Parameters:
    - train_data (pd.DataFrame): The training data.
    - test_data (pd.DataFrame): The testing data.
    - max_features (int): The max number of features to consider for TF-IDF.
    
    Returns:
    - (pd.DataFrame, pd.DataFrame): The transformed training and testing data as a tuple
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values
        
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
        train_df['target'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
        test_df['target'] = y_test
        
        
        
        logger.debug("TF-IDF transformation applied successfully")
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error applying TF-IDF transformation: {e}")
        raise
    
    
def save_data(data: pd.DataFrame, file_path: str):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame to save.
    - file_path (str): The path where the DataFrame will be saved.
    """
    try:
        data.to_csv(file_path, index=False)
        logger.debug(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise    

def main():
    """
    Main function to run the feature engineering module.
    It will load the data, apply TF-IDF transformation, and save the processed data.
    """
    try:
        train_data_path = './data/interim/train_preprocessed.csv'  # Path to the preprocessed training data
        test_data_path = './data/interim/test_preprocessed.csv'    # Path to the preprocessed testing data  
        max_features = 50  # Max number of features for TF-IDF
        
        train_data = load_data(train_data_path)  # Load training data
        test_data = load_data(test_data_path)    # Load testing data
        
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)  # Apply TF-IDF transformation
        
        
        data_save_path = './data/processed'  # Path to save the final transformed data
        os.makedirs(data_save_path, exist_ok=True)  # Ensure the directory exists
        
        save_data(train_df, os.path.join(data_save_path, 'train_tfidf.csv'))  # Save transformed training data
        save_data(test_df, os.path.join(data_save_path, 'test_tfidf.csv'))    # Save transformed testing data
        
    except Exception as e:
        logger.error(f"Failed to complete the feature engineering: {e}")
        raise   
    
    
    
if __name__ == "__main__":
    main()  # Run the main function to execute the feature engineering module