from flask import Flask, render_template, request
import pandas as pd
import faiss
import numpy as np
import json
import openai
from sentence_transformers import SentenceTransformer
import os

# Initialize Flask app
app = Flask(__name__)

# OpenAI API key setup (replace with your API key)
openai.api_key = "your-openai-api-key"  # <-- Replace with your OpenAI API key

# Load CSV data
csv_path = 'who_outbreaks.csv'  # <-- Replace with your correct CSV file path
df = pd.read_csv(csv_path)

# Load FAISS index
index_path = 'who_faiss_index.bin'  # <-- Replace with your correct FAISS index file path
faiss_index = faiss.read_index(index_path)  # <-- use the correct FAISS index file

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to parse 'regions' column correctly
def parse_regions(regions_field):
    if pd.isna(regions_field):
        return "Global"
    try:
        cleaned = regions_field.replace("'", '"').strip()
        regions = json.loads(cleaned)
        if isinstance(regions, list):
            return ", ".join(regions)
        return str(regions)
    except Exception:
        return str(regions_field)

# Function to search outbreak data
def search_outbreaks(query, k=5):
    """Search FAISS and get outbreak records"""
    print(f"🔍 Searching outbreak records for query: {query}")

    query_embedding = model.encode([query])
    distances, ids = faiss_index.search(np.array(query_embedding).astype('float32'), k)

    clean_ids = [int(id) for id in ids[0]]
    if not clean_ids or all(id == -1 for id in clean_ids):
        print("No matches found.")
        return []

    results = []
    for id in clean_ids:
        if id < 0:
            continue
        match = df[df['id'] == id]  # Fix to match using 'id'
        if not match.empty:
            row = match.iloc[0]
            result = {
                "id": row['id'],
                "title": row['title'],
                "date": row['date'],
                "regions": parse_regions(row['regions']),
                "summary": row['summary'],
                "url": row['url'],
                "source": row['source']
            }
            print(f"Found match: {result['title']} ({result['date']})")
            results.append(result)

    return results

# Function to generate WHO insights using OpenAI API
def generate_who_insights(context, question):
    """Generate insights using OpenAI's GPT model with error handling"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a WHO public health analyst. Provide concise insights based on outbreak data."
            }, {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }],
            temperature=0.3,
            max_tokens=200
        )
        return response['choices'][0]['message']['content']
    except openai.OpenAIError as e:
        return f"Analysis failed: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Flask route
@app.route('/', methods=['GET', 'POST'])
def home():
    results = []
    insights = ""
    query = ""

    if request.method == 'POST':
        query = request.form['query']
        results = search_outbreaks(query)
        
        if results:
            context = "\n\n".join(
                f"Title: {doc['title']}\nDate: {doc['date']}\nRegions: {doc['regions']}\nSummary: {doc['summary']}"
                for doc in results
            )
            insights = generate_who_insights(context, query)

    return render_template('index.html', query=query, results=results, insights=insights)

if __name__ == '__main__':
    app.run(debug=True)
