from flask import Flask, render_template,request,redirect, url_for,session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import math
import re
import os
import nltk
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import Levenshtein
import json

app = Flask(__name__)
app.secret_key = '07'
def preprocess_text(text):
    # Remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
    # Convert to lowercase
    clean_text = clean_text.lower()
    return clean_text

# def load_inverted_index(file_path):
#     inverted_index = {}
#     with open(file_path, 'r') as file:
#         inverted_index_json = json.load(file)
    
#     return inverted_index_json
# Load the inverted index from a text file

def retrieve_posting_list_from_disk(root, term):
    current_node = root
    for char in term:
        current_path = os.path.join(current_node, char)
        if not os.path.exists(current_path):
            return None
        current_node = current_path

    posting_list_file = os.path.join(current_node, "posting_list.json")
    if os.path.exists(posting_list_file):
        with open(posting_list_file, "r") as f:
            return json.load(f)

    return None

def load_inverted_index(root, query):
    inverted_index = {}
    
    for term_to_search in query.split():
        posting_list = retrieve_posting_list_from_disk(root, term_to_search)
        if posting_list is not None:
            inverted_index[term_to_search] = posting_list

    return inverted_index

# Calculate TF-IDF scores
def calculate_tfidf(inverted_index):
    doc_term_matrix = defaultdict(lambda: defaultdict(int))
    term_document_count = defaultdict(int)
    total_documents = len(inverted_index)

    for term, postings in inverted_index.items():
        for doc_id, position in postings:
            doc_term_matrix[doc_id][term] = 1  # Using binary term frequency
            term_document_count[term] += 1

    idf = {term: math.log(total_documents / (1 + term_document_count[term])) for term in inverted_index}

    tfidf_matrix = {}
    for doc_id, term_counts in doc_term_matrix.items():
        tfidf_matrix[doc_id] = {term: tf * idf[term] for term, tf in term_counts.items()}

    return tfidf_matrix

# Rank documents based on TF-IDF similarity to the query
def rank_documents(query, inverted_index, tfidf_matrix, batch_size=1000,top_k=10):
    query = preprocess_text(query)
    query_vectorizer = TfidfVectorizer()
    query_tfidf = query_vectorizer.fit_transform([query])

    ranked_documents = {}

    for doc_id, term_tfidf in tfidf_matrix.items():
        doc_tfidf = [term_tfidf.get(term, 0) for term in query_vectorizer.get_feature_names_out()]
        similarity = cosine_similarity(query_tfidf, [doc_tfidf])
        ranked_documents[doc_id] = similarity[0, 0]

    ranked_documents = sorted(ranked_documents.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return ranked_documents

# def rank_documents_with_feedback(query, inverted_index, tfidf_matrix, batch_size=1000, top_k=10):
#     # Step 1: Preprocess the query
#     query = preprocess_text(query)
#     user_feedback = session.get('user_feedback', {})

#     # Print user feedback for debugging
#     print("User Feedback:", user_feedback)

#     # Step 2: Find documents containing query terms in inverted index
#     relevant_docs = set()
#     for term in query.split():
#         if term in inverted_index:
#             relevant_docs.update([doc_id for doc_id, _ in inverted_index[term]])

#     # Step 3: Calculate TF-IDF vector for the query
#     query_vectorizer = TfidfVectorizer()
#     query_tfidf = query_vectorizer.fit_transform([query])

    
#     # Step 4: Batch process relevant documents
#     ranked_documents_dict = {}  # Initialize a dictionary to store document scores
#     for doc_id in relevant_docs:
#         term_tfidf = tfidf_matrix.get(doc_id, {})
#         doc_tfidf = [term_tfidf.get(term, 0) for term in query_vectorizer.get_feature_names_out()]
#         similarity = cosine_similarity(query_tfidf, [doc_tfidf])

#         # Use cosine similarity as the initial score
#         ranked_documents_dict[doc_id] = similarity[0, 0]

#     # Adjust the ranking based on user feedback
#     for doc_id, feedback_data in user_feedback.items():
#         if isinstance(feedback_data, dict):
#             user_query = feedback_data.get('query', '')
#             feedback = feedback_data.get('feedback', '')

#             print(f"Doc ID: {doc_id}, User Query: {user_query}, Feedback: {feedback}")

#             if user_query and doc_id in relevant_docs:
#                 if doc_id not in ranked_documents_dict:
#                     ranked_documents_dict[doc_id] = 0  # Initialize score if not already present

#                 # Increase the score for relevant documents, decrease for non-relevant
#                 adjustment = 0.2 if feedback == 'relevant' else -0.2
#                 ranked_documents_dict[doc_id] += adjustment

#     # Sort documents based on the adjusted scores
#     ranked_documents = sorted(ranked_documents_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

#     return ranked_documents

# inverted_index_file_path = 'try2.txt'  # Replace with the actual file path
# inverted_index = load_inverted_index(inverted_index_file_path)

# tfidf_matrix = calculate_tfidf(inverted_index)


# ranked_documents = rank_documents_using_tfidf(user_query, inverted_index, tfidf_matrix)

def load_documents(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            doc_id = filename.split('.')[0]  # Extract 'd4' from 'd4.txt'
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                title_start = content.find("Title:") + len("Title:")
                title_end = content.find("Paragraph:")
                title = content[title_start:title_end].strip()

                paragraph_start = title_end + len("Paragraph:")
                paragraph = content[paragraph_start:].strip()

                documents[doc_id] = {'title': title, 'paragraph': paragraph}

    return documents

def correct_query(query):
    dictionary = set(nltk.corpus.words.words())
    words = word_tokenize(query)

    corrected_words = [word if word in dictionary else suggest_correction(word, dictionary) for word in words]

    corrected_query = ' '.join(corrected_words)

    return corrected_query

def suggest_correction(word, dictionary):
    # Calculate Levenshtein distance for each word in the dictionary
    distances = [(dict_word, Levenshtein.distance(word, dict_word)) for dict_word in dictionary]

    # Sort by distance and suggest the closest word
    suggested_word = min(distances, key=lambda x: x[1])[0]

    return suggested_word

def get_doc_id(term, inverted_index):
    for _, postings in inverted_index.items():
        for doc_id, _ in postings:
            if doc_id:
                return doc_id

    return None
user_feedback = {}

@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        query = request.form.get('query')
        doc_id = request.form.get('doc_id')
        feedback = request.form.get('feedback')
        
        session.setdefault('user_feedback', {})
        # Store the feedback in the user_feedback dictionary
        session['user_feedback'][doc_id] = {'query': query, 'feedback': feedback}

        # You may want to update your ranking model based on the feedback here
        # For simplicity, we'll just print the feedback for now
        print(f"Feedback for document {doc_id}: {feedback}")

    # Redirect back to the search page
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')

        
        stop_words = set(stopwords.words('english'))
        query_tokens = nltk.word_tokenize(query)
        filtered_query_tokens = [word for word in query_tokens if word.lower() not in stop_words]
        filtered_query = ' '.join(filtered_query_tokens)

        inverted_index = load_inverted_index('root',filtered_query)
        tfidf_matrix = calculate_tfidf(inverted_index)

        ranked_documents = rank_documents(filtered_query, inverted_index, tfidf_matrix, batch_size=1000)
        # ranked_documents = rank_documents_with_feedback(filtered_query, inverted_index, tfidf_matrix, batch_size=1000)
        corrected_query = correct_query(query)
        # Extract title and 2-3 sentences for each document
        documents_directory = 'All_textfiles'
        documents = load_documents(documents_directory)
        result_documents = []
        for doc_id, _ in ranked_documents:
            title = documents[doc_id]['title']
            paragraph = documents[doc_id]['paragraph'][:200]  # Adjust as needed
            result_documents.append({'title': title, 'paragraph': paragraph})
        # print(result_documents)   
        if corrected_query != query:
           suggestion = suggest_correction(query, set(nltk.corpus.words.words()))
        else:
           suggestion = None
            
        return render_template('index.html', inverted_index=inverted_index, get_doc_id=get_doc_id, query=query, corrected_query=corrected_query,result_documents=result_documents,suggestion=suggestion)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)