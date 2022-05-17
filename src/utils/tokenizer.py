import logging
import string
import pandas as pd

from logging import error as error
from logging import exception as exception
from logging import info as info
from logging import warning as warning

import nltk
import numpy as np
# nltk.download('punkt')
import tqdm
from tqdm import tqdm

# logging.basicConfig(filename='tokenization.log', level=logging.INFO)
info = print

class Tokenizer:

    def __init__(
        self,
        ngrams=None,
        is_char_tokenization=None,
        text_cleaning_method=None,
        return_type='np.array'
    ):

        self.ngrams = ngrams
        self.is_char_tokenization = is_char_tokenization
        self.text_cleaning_method = text_cleaning_method
        self.return_type = return_type
        
        info("Tokenization initialized with.. ")
        info("- Q-gramms: ", self.ngrams)
        info("- Char-Tokenization: ", self.is_char_tokenization)
        info("- Text text_cleaning_methodning process: ", self.text_cleaning_method)
        
    def process(self, input, columns=None) -> np.array:

        if isinstance(input, pd.DataFrame):
            input_df = input.copy()

            if columns != None and len(columns) > 0:
                input_df = input_df[columns]

            data = input_df['merged'] = input_df.apply(" ".join, axis=1)
        else:
            data = input


        self.data_size = len(data)
        self.data = np.array(data, dtype=object)
        self.tokenized_data = np.empty([self.data_size], dtype=object)
        
        info("\nProcessing strarts.. ")
        info("- Initial data size: ", self.data_size)

        if self.text_cleaning_method is not None or self.ngrams or self.is_char_tokenization:
            for i in tqdm(range(0, self.data_size), desc="Processing.."):
                if self.text_cleaning_method is not None:
                    record = self.text_cleaning_method(self.data[i])
                else:
                    record = self.data[i]

                if self.is_char_tokenization is not None:
                    if self.is_char_tokenization:
                        record = set(nltk.ngrams(record, n=self.ngrams))
                    else:
                        if len(nltk.word_tokenize(record)) > self.ngrams:
                            record = set(nltk.ngrams(nltk.word_tokenize(record), n=self.ngrams))
                        else:
                            record = set(nltk.ngrams(nltk.word_tokenize(record), n=len(nltk.word_tokenize(string))))
                    print(record)

                self.tokenized_data[i] = record
                
        else:
            self.tokenized_data = self.data

        if self.return_type == 'list':
            return self.tokenized_data.tolist()

        return self.tokenized_data

def cora_text_cleaning_method(s):

    new_s = s

    # Lower letters
    new_s = new_s.lower()

    # # Remove special chars
    # new_s = new_s.replace("\n", "").replace("/z", " ")

    # Remove pancutation
    new_s = new_s.translate(str.maketrans('', '', string.punctuation))

    return str(new_s)
