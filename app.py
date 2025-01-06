import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from textblob import TextBlob

def load_paper(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def extract_keywords(text, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    return keywords

def assess_readability(text):
    return textstat.flesch_reading_ease(text)

def grammar_sentiment_analysis(text):
    blob = TextBlob(text)
    errors = sum(1 for sentence in blob.sentences if len(sentence.correct()) != len(sentence))
    sentiment = blob.sentiment.polarity
    return errors, sentiment

def evaluate_alignment(keywords, known_topics):
    overlap = set(keywords).intersection(set(known_topics))
    alignment_score = len(overlap) / len(known_topics)
    return alignment_score, list(overlap)

def assess_publishability(content, known_topics):
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

st.title("Research Paper Publishability Assessment Tool")

uploaded_file = st.file_uploader("Upload your research paper (TXT format only):", type=["txt"])

known_topics = st.text_input("Enter known research topics (comma-separated):", "language models, chain of thought, interpretability, Bayesian inference")

if uploaded_file and known_topics:
    known_topics_list = [topic.strip() for topic in known_topics.split(",")]
    paper_content = uploaded_file.read().decode("utf-8")

    st.write("Processing your document...")
    report = assess_publishability(paper_content, known_topics_list)

    st.subheader("Publishability Report")
    st.write(f"**Title:** {report['Title']}")
    st.write(f"**Keywords:** {', '.join(report['Keywords'])}")
    st.write(f"**Readability Score (Flesch):** {report['Readability Score (Flesch)']:.2f}")
    st.write(f"**Grammar Errors:** {report['Grammar Errors']}")
    st.write(f"**Sentiment Polarity:** {report['Sentiment Polarity']:.2f}")
    st.write(f"**Alignment Score:** {report['Alignment Score']:.2f}")
    st.write(f"**Aligned Keywords:** {', '.join(report['Aligned Keywords'])}")

else:
    st.write("Please upload a TXT file and provide known research topics.")
