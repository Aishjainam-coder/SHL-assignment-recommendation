import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import json
import os
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page config for better appearance
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="📝", layout="centered")

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
        margin-top: 1em;
    }
    .stTextArea textarea {
        border-radius: 8px;
        min-height: 100px;
    }
    .stDataFrame {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Utility Functions ---
def normalize_text(text):
    return ' '.join(text.lower().strip().split())

def normalize_url(url):
    return url.lower().strip().rstrip('/') if url else ''

# --- SHLRecommender Class ---
class SHLRecommender:
    def __init__(self, data_path: str = "data/shl_assessments.json"):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        self.prepare_embeddings()

    def prepare_embeddings(self):
        assessment_texts = [
            normalize_text(
                f"{a['name']} {a['test_type']} {a['duration']} {a.get('description','')} {a.get('skills','')}"
            )
            for a in self.assessments
        ]
        self.embeddings = self.model.encode(assessment_texts)

    def extract_duration(self, text: str):
        import re
        match = re.search(r'(\d+)\s*(?:min|minute|mins|minutes)', text.lower())
        return int(match.group(1)) if match else None

    def recommend(self, query, max_duration=None, max_results=10, similarity_threshold=0.25):
        query = normalize_text(query)
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= max_results:
                break
            assessment = self.assessments[idx]
            duration = self.extract_duration(assessment['duration'])
            if max_duration and duration and duration > max_duration:
                continue
            if similarities[idx] >= similarity_threshold:
                results.append({
                    "assessment_name": assessment['name'],
                    "assessment_url": normalize_url(assessment['url']),
                    "remote_testing_support": assessment['remote_testing_support'],
                    "adaptive_irt_support": assessment['adaptive_irt_support'],
                    "duration": assessment['duration'],
                    "test_type": assessment['test_type'],
                    "similarity_score": float(similarities[idx])
                })
        # If no results, print and return empty
        if not results:
            print("No recommendations found.")
        return results

# --- Evaluation Pipeline ---
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / len(relevant) if relevant else 0.0

def average_precision_at_k(recommended, relevant, k):
    score = 0.0
    num_hits = 0.0
    for i, rec in enumerate(recommended[:k]):
        if rec in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0

def evaluate_recommender(recommender, test_set, k=3):
    recalls = []
    aps = []
    for test in test_set:
        query = test["query"]
        relevant_urls = set([normalize_url(url) for url in test["relevant_urls"]])
        results = recommender.recommend(query, max_results=k)
        recommended_urls = [normalize_url(r["assessment_url"]) for r in results]
        print("Recommended URLs:", recommended_urls)
        print("Relevant URLs:", relevant_urls)
        recall = recall_at_k(recommended_urls, relevant_urls, k)
        ap = average_precision_at_k(recommended_urls, relevant_urls, k)
        recalls.append(recall)
        aps.append(ap)
        print(f"Query: {query}\nRecall@{k}: {recall:.2f}, AP@{k}: {ap:.2f}\n")
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    map_k = sum(aps) / len(aps) if aps else 0.0
    print(f"Mean Recall@{k}: {mean_recall:.3f}")
    print(f"MAP@{k}: {map_k:.3f}")
    return mean_recall, map_k

# --- Example Test Set (add more as needed) ---
test_set = [
    {
        "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "relevant_urls": [
            "https://www.shl.com/solutions/products/product-catalog/view/automata-fix-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/",
            "https://www.shl.com/solutions/products/product-catalog/view/agile-software-development/",
            "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/",
            "https://www.shl.com/solutions/products/product-catalog/view/computer-science-new/"
        ]
    },
    # Add more test cases here
]

# Cache the recommender to avoid reloading on every rerun
@st.cache_resource(show_spinner=True)
def get_recommender():
    data_path = "data/shl_assessments.json"
    if not verify_data_file(data_path):
        st.error("Data file verification failed. Please check the data file.")
        return None
    return SHLRecommender(data_path)

def clean_duration(duration):
    # Only allow numbers and the word 'minute'
    import re
    match = re.search(r'(\\d+).*min', str(duration).lower())
    if match:
        return f"{match.group(1)} minutes"
    # fallback if not found
    return "N/A"

def main():
    st.sidebar.image("https://www.shl.com/wp-content/themes/shl/images/logo.svg", width=180)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app recommends the most relevant SHL assessments for your hiring needs.\n
        - Enter a job description or requirements.\n
        - Set a maximum duration (optional).\n
        - Get up to 10 recommended assessments with key details.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Built with ❤️ using Streamlit")

    st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>SHL Assessment Recommender</h1>
    <p style='text-align: center; color: #333; font-size: 1.1em;'>
        Enter your job description or requirements to get recommended SHL assessments.
    </p>
    """, unsafe_allow_html=True)

    with st.form("recommendation_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_area(
                "Enter job description or requirements:",
                placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams..."
            )
        with col2:
            max_duration = st.number_input(
                "Maximum duration (minutes):",
                min_value=0,
                value=60,
                help="Set to 0 for no duration limit"
            )
        submit = st.form_submit_button("Get Recommendations")

    if submit and query:
        with st.spinner("🔎 Getting recommendations..."):
            try:
                recommender = get_recommender()
                if recommender is None:
                    st.error("Failed to initialize the recommender. Please try again later.")
                    return
                    
                recs = recommender.recommend(
                    query, 
                    max_duration if max_duration > 0 else None
                )
                
                st.markdown("---")
                if not recs:
                    st.warning("No recommendations found. Please try a different query or use more specific keywords.")
                else:
                    st.success(f"Found {len(recs)} recommendations:")
                    
                    # Create a DataFrame with sorted recommendations
                    df = pd.DataFrame(recs)
                    df = df.sort_values('similarity_score', ascending=False)
                    
                    # Display the DataFrame
                    st.dataframe(df, use_container_width=True)
                    
                    # Show detailed view
                    with st.expander("Show detailed recommendations"):
                        for idx, rec in enumerate(recs, 1):
                            st.markdown(f"### Recommendation {idx}")
                            st.markdown(f"**Assessment Name:** {rec['assessment_name']}")
                            st.markdown(f"**Similarity Score:** {rec['similarity_score']:.3f}")
                            st.markdown(f"**Test Type:** {rec['test_type']}")
                            duration = clean_duration(rec.get('duration', ''))
                            st.markdown(f"**Duration:** {duration}")
                            st.markdown(f"**Remote Testing Support:** {rec['remote_testing_support']}")
                            st.markdown(f"**Adaptive IRT Support:** {rec['adaptive_irt_support']}")
                            st.markdown(f"[Assessment URL]({rec['assessment_url']})")
                            st.markdown("---")
            except Exception as e:
                logger.error(f"Error getting recommendations: {str(e)}")
                st.error(f"An error occurred while getting recommendations: {str(e)}")

app = FastAPI()

class RecommendationRequest(BaseModel):
    query: str
    max_duration: Optional[int] = None

class Assessment(BaseModel):
    assessment_name: str
    assessment_url: str
    remote_testing_support: str
    adaptive_irt_support: str
    duration: str
    test_type: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=List[Assessment])
def recommend(request: RecommendationRequest):
    # Call your SHLRecommender here and return the results
    # Example:
    # results = recommender.recommend(request.query, request.max_duration)
    # return results
    pass

# Optionally, add a Streamlit button to run evaluation from the UI
if 'st' in globals():
    with st.sidebar:
        if st.button('Run Evaluation Pipeline'):
            st.info('Running evaluation...')
            recommender = get_recommender()
            if recommender:
                mean_recall, map_k = evaluate_recommender(recommender, test_set, k=3)
                st.success(f"Mean Recall@3: {mean_recall:.3f}\nMAP@3: {map_k:.3f}")
            else:
                st.error('Recommender not initialized.')

if __name__ == "__main__":
    main() 
