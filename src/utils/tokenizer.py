import logging
import string
from logging import error as error
from logging import exception as exception
from logging import info as info
from logging import warning as warning

import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
from tqdm import tqdm

logging.basicConfig(filename='tokenization.log', level=logging.INFO)
info = print

class Tokenizer:
     
    
    def __init__(
        self,
        ngrams = None, 
        is_char_tokenization = None, 
        clean = None,
        return_type = 'list'  
    ):
        
        self.ngrams = ngrams
        self.is_char_tokenization = is_char_tokenization
        self.clean = clean
        self.return_type = return_type
        
        info("Tokenization initialized with.. ")
        info("- Q-gramms: ", self.ngrams)
        info("- Char-Tokenization: ", self.is_char_tokenization)
        info("- Text cleanning process: ", self.clean)
        
    def process(self, data):
        
        # if isinstance(data, list):
        # elif isinstance(data, pd.DataFrame):
        # elif isinstance(data, np.array):
            
        self.data_size = len(data)
        self.data = np.array(data, dtype = object)
        self.tokenized_data = np.empty([self.data_size], dtype = object)
        
        info("\nProcessing strarts.. ")
        info("- Data size: ", self.data_size)
        
        # self.data_mapping = np.array(input_strings, dtype=object)
        
        for i in tqdm(range(0, self.data_size), desc="Processing.."):
            if self.clean is not None:
                string = self.clean(self.data[i])
            else:
                string = self.data[i]
            # info(string)
            if self.is_char_tokenization:
                self.tokenized_data[i] = set(nltk.ngrams(string, n = self.ngrams))
            else:
                if len(nltk.word_tokenize(string)) > self.ngrams:
                    self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = self.ngrams))
                else:
                    self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = len(nltk.word_tokenize(string))))
            # info(self.tokenized_data[i])

        if self.return_type == 'list':
            return self.tokenized_data.tolist()

        return self.tokenized_data

    # def process_df(self, df):
        
    #     # if isinstance(data, list):
    #     # elif isinstance(data, pd.DataFrame):
    #     # elif isinstance(data, np.array):
    #     for col in df:
    #         data = df[col]
    #         self.data_size = len(data)
    #         self.data = np.array(data, dtype = object)
    #         self.tokenized_data = np.empty([self.data_size], dtype = object)
            
    #         info("\nProcessing strarts.. ")
    #         info("- Data size: ", self.data_size)
            
    #         # self.data_mapping = np.array(input_strings, dtype=object)
            
    #         for i in tqdm(range(0, self.data_size), desc="Processing.."):
    #             if self.clean is not None:
    #                 string = self.clean(self.data[i])
    #             else:
    #                 string = self.data[i]
    #             # info(string)
    #             if self.is_char_tokenization:
    #                 self.tokenized_data[i] = set(nltk.ngrams(string, n = self.ngrams))
    #             else:
    #                 if len(nltk.word_tokenize(string)) > self.ngrams:
    #                     self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = self.ngrams))
    #                 else:
    #                     self.tokenized_data[i] = set(nltk.ngrams(nltk.word_tokenize(string), n = len(nltk.word_tokenize(string))))
    #             # info(self.tokenized_data[i])

    #         df[col] =

    #     return df


def clean(s):
    
    new_s = " ".join(s)

    # Lower letters 
    new_s = new_s.lower()
    
    # Remove unwanted chars 
    new_s = new_s.replace("\n", " ").replace("/z", " ")
    
    # Remove pancutation     
    new_s = new_s.translate(str.maketrans('', '', string.punctuation))
    
    return str(new_s)
