import os
import re
import string
from flask import Flask, render_template, request
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load custom Hindi stopwords
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().split(',')
    return set([word.strip() for word in stopwords])

# Preprocess text: remove stopwords, punctuation, etc.
def preprocess_text(text, stopwords):
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = text.lower()
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return " ".join(cleaned_words)

# Read documents from the hindi_docs folder
def load_documents(documents_folder):
    documents = []
    for filename in os.listdir(documents_folder):
        filepath = os.path.join(documents_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

# Main function to perform cross-language information retrieval
def clir_system(query, documents_folder='hindi_docs/', stopwords_file='hindi_stopwords.txt'):
    hindi_stopwords = load_stopwords(stopwords_file)
    documents = load_documents(documents_folder)
    preprocessed_docs = [preprocess_text(doc, hindi_stopwords) for doc in documents]

    translator = Translator()
    translated_query = translator.translate(query, src='en', dest='hi').text
    preprocessed_query = preprocess_text(translated_query, hindi_stopwords)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    query_vector = vectorizer.transform([preprocessed_query])
    
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]  # Sort in descending order
    
    results = []
    for idx in ranked_indices:
        results.append({
            'document': f'Document {idx + 1}',
            'similarity_score': cosine_sim[idx]
        })
    
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        results = clir_system(query)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
