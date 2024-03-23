import os
import re
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# Download the NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_document(document):
    # Convert to lowercase
    document = document.lower()

    # Remove punctuation
    document = re.sub(r'[^\w\s]', '', document)

    return document

def create_inverted_index(folder_path):
    inverted_index = defaultdict(list)
    stop_words = set(stopwords.words('english'))
    stop_words.update(["title", "paragraph"])
    stemmer = PorterStemmer()

    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            document = file.read()

            # Preprocess the document
            document = preprocess_document(document)

            # Tokenize
            tokens = [token for token in word_tokenize(document) if token.isalnum() and token not in stop_words]

            # Remove ".txt" extension
            file_name_without_extension = os.path.splitext(filename)[0]

            for position, term in enumerate(tokens):
                inverted_index[term].append((file_name_without_extension, position + 1))

    return inverted_index

# Set your folder path and output file
folder_path = 'All_textfiles'
output_file = 'index.txt'

# Create the inverted index
inverted_index = create_inverted_index(folder_path)

# Save the inverted index to a text file
with open(output_file, 'w', encoding='utf-8') as out_file:
    for term, positions in inverted_index.items():
        out_file.write(f"{term}: {positions}\n")
