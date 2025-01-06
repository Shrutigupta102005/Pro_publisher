import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
from textblob import TextBlob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Load Paper Content

def load_paper(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Extract Keywords Using TF-IDF

def extract_keywords(text, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    return keywords

# Check Readability

def assess_readability(text):
    return textstat.flesch_reading_ease(text)

# Grammar and Sentiment Analysis

def grammar_sentiment_analysis(text):
    blob = TextBlob(text)
    errors = sum(1 for sentence in blob.sentences if len(sentence.correct()) != len(sentence))
    sentiment = blob.sentiment.polarity
    return errors, sentiment

# Evaluate Research Trends Alignment

def evaluate_alignment(keywords, known_topics):
    overlap = set(keywords).intersection(set(known_topics))
    alignment_score = len(overlap) / len(known_topics)
    return alignment_score, list(overlap)

# Main Function for Publishability Assessment

def assess_publishability(filepath, known_topics):
    content = load_paper(filepath)
    
    # Extract Title and Abstract
    title_match = re.search(r'(\bAbstract\b.*?\n)(.*?\n\n)', content, re.DOTALL)
    title = content.split("\n")[0].strip()
    abstract = title_match.group(2).strip() if title_match else "Abstract not found"
    
    # Step 1: Keyword Extraction
    keywords = extract_keywords(abstract)
    
    # Step 2: Readability Analysis
    readability_score = assess_readability(content)
    
    # Step 3: Grammar and Sentiment
    grammar_errors, sentiment = grammar_sentiment_analysis(content)
    
    # Step 4: Alignment with Research Trends
    alignment_score, aligned_keywords = evaluate_alignment(keywords, known_topics)

    # Summary Report
    report = {
        "Title": title,
        "Keywords": keywords,
        "Readability Score (Flesch)": readability_score,
        "Grammar Errors": grammar_errors,
        "Sentiment Polarity": sentiment,
        "Alignment Score": alignment_score,
        "Aligned Keywords": aligned_keywords
    }

    return report

# Visualize Report

def visualize_report(report):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Readability Score (Flesch)", "Alignment Score"]
    values = [report["Readability Score (Flesch)"], report["Alignment Score"] * 100]
    
    ax.bar(metrics, values, color=['blue', 'green'])
    ax.set_ylabel('Score')
    ax.set_title('Publishability Assessment Metrics')
    plt.show()

# Example Usage

if __name__ == "__main__":
    paper_path = "example_paper.txt"  # Replace with the actual path
    known_topics = ["language models", "chain of thought", "interpretability", "Bayesian inference"]
    
    report = assess_publishability(paper_path, known_topics)
    print("Publishability Report:")
    for key, value in report.items():
        print(f"{key}: {value}")

    visualize_report(report)
