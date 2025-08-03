import pandas as pd
import os
import logging

from sklearn.preprocessing import LabelEncoder 
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string


nltk.download('stopwords', quiet=True)  # Download stopwords if not already downloaded
nltk.download('punkt', quiet=True)  # Download punkt tokenizer if not already downloaded



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

logger = logging.getLogger('data_preprocessing') #This will create a logger named 'data_ingestion'
logger.setLevel(logging.DEBUG) #Set the logging level to INFO


#defining the console handler
console_handler = logging.StreamHandler() #This will create a console handler
console_handler.setLevel(logging.DEBUG) #Set the logging level for the console handler to INFO


#Definging the file handler
log_file_path = os.path.join(log_dir, 'data_preprocessing.log') #Define the path for the log file
file_handler = logging.FileHandler(log_file_path) #This will create a file handler
file_handler.setLevel(logging.DEBUG) #Set the logging level for the file handler to DEBUG

#Define the format of the log messages, below format will include the timestamp, logger name, log level, and the actual log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #This will format the log messages

console_handler.setFormatter(formatter) #Set the formatter for the console handler
file_handler.setFormatter(formatter) #Set the formatter for the file handler

#Add the handlers to the logger
logger.addHandler(console_handler) #Add the console handler to the logger
logger.addHandler(file_handler) #Add the file handler to the logger


#------------------------------------------------------ DATA TRANSFORMATION FUNCTION ------------------------------------------------------##This code is used to preprocess the data, including text cleaning and encoding categorical variables


def transform_text(text: str) -> str:
    """
    Clean and preprocess the input text.
    The preprocessing for messages column includes:
    - Converting to lowercase
    - Tokenization
    - Removing punctuation
    - Removing stopwords
    - Stemming the words
    
    Parameters:
    - text (str): The text to be cleaned and preprocessed.

    
    Returns:
    - str: The cleaned and preprocessed text.
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    
    #Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    
    #Removing punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Stem the words
    porter_stemmer = PorterStemmer()
    text = [porter_stemmer.stem(word) for word in text]
    
    cleaned_text = ' '.join(text)
    
    return cleaned_text



#------------------------------------------------------ MAIN PREPROCESSING FUNCTION ------------------------------------------------------##This code is the main preprocessing function that will be called to preprocess the data

def preprocess_data(data: pd.DataFrame, text_column: str = "text", target_column: str = "target") -> pd.DataFrame:
    """
    Preprocess the data by cleaning text and encoding categorical variables.
    
    Parameters:
    - data (pd.DataFrame): The dataset to be preprocessed.
    
    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """
    try:
        logger.debug("Starting data preprocessing")
        
        #Encode trhe target variable
        label_encoder = LabelEncoder()
        data[target_column] = label_encoder.fit_transform(data[target_column])
        logger.debug("Encoded target variable")
        
        #Remove duplicate rows
        data = data.drop_duplicates(keep='first')
        logger.debug("Removed duplicate rows")
        
        #Apply text transformation
        data.loc[:,text_column] = data[text_column].apply(transform_text)
        logger.debug("Transformed text data")
        
        return data
    
    except KeyError as e:
        logger.error(f"Missing Columns in Data frame: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise
    
    
#------------------------------------------------------ Defining thr main function ------------------------------------------------------##This code is used to run the data preprocessing module
 
def main():
    """
    Main function to run the data preprocessing module.
    It will load the data, preprocess it, and save the processed data.
    """
try:
    train_data_path = './data/raw/train.csv'  # Path to the training data
    test_data_path = './data/raw/test.csv'    # Path to the testing data  
    data_path = './data/interim'  # Path to save the processed data
    
    
    
    train_data = pd.read_csv(train_data_path)  # Load training data
    test_data = pd.read_csv(test_data_path)    # Load testing data
    
    logger.debug("Data loaded successfully")  # Log successful data loading
    
    # Train and test data will be preprocessed seperately and saved 
    train_data_preprocessed = preprocess_data(train_data)  # Preprocess the training data
    test_data_preprocessed = preprocess_data(test_data)    # Preprocess the testing data
    
    
    #Save the preprocessed data
    os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists
    train_data_preprocessed.to_csv(os.path.join(data_path, 'train_preprocessed.csv'), index=False)  # Save preprocessed training data
    test_data_preprocessed.to_csv(os.path.join(data_path, 'test_preprocessed.csv'), index=False)    # Save preprocessed testing data
    
    
    logger.debug(f"Data saved successfully in {data_path}")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")

except pd.errors.EmptyDataError as e:
    logger.error(f"Empty data file: {e}")
except Exception as e:
    logger.error(f"Failed to complete the data transformation process: {e}")
    raise
    
    
if __name__ == "__main__":
    main()  # Run the main function to execute the data preprocessing module
  