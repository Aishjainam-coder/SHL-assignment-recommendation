import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import json
import os
import logging
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page config for better appearance
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="📊",
    layout="wide"
)

# Constants
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.3
TOP_K = 5

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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# --- Utility Functions ---
def normalize_text(text):
    return ' '.join(text.lower().strip().split())

def normalize_url(url):
    return url.lower().strip().rstrip('/') if url else ''

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

# --- Example Test Set ---
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
    }
]

class SHLRecommender:
    def __init__(self, data_path="data/shl_assessments.csv"):
        self.data_path = data_path
        self.model = None
        self.data = None
        self.embeddings = None
        self._load_data()
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            if self.model is None:
                self.model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _load_data(self):
        """Load and preprocess the assessment data."""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            self.data['combined_text'] = self.data.apply(
                lambda row: ' '.join(str(val) for val in row if pd.notna(val)), 
                axis=1
            )
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _preprocess_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        try:
            if self.embeddings is None:
                self.embeddings = self.model.encode(texts, show_progress_bar=True)
            return self.embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
            
    def recommend(self, query, max_duration=None, similarity_threshold=0.3, max_results=10):
        """Get assessment recommendations based on the query."""
        try:
            # Preprocess query
            query = self._preprocess_text(query)
            if not query:
                return []
            
            # Generate embeddings
            embeddings = self._get_embeddings(self.data['combined_text'].tolist())
            
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][:max_results]
            recommendations = []
            
            for idx in top_indices:
                if similarities[idx] >= similarity_threshold:
                    rec = {
                        'Assessment': self.data.iloc[idx]['Assessment'],
                        'Category': self.data.iloc[idx]['Category'],
                        'Description': self.data.iloc[idx]['Description'],
                        'Similarity Score': similarities[idx]
                    }
                    
                    # Add duration filter if specified
                    if max_duration is not None:
                        duration = self.data.iloc[idx].get('Duration', 0)
                        if duration and int(duration) <= max_duration:
                            recommendations.append(rec)
                    else:
                        recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

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
    <h1 style='text-align: center; color: #4F8BF9;'>SHL Assessment Recommendation System</h1>
    <p style='text-align: center; color: #333; font-size: 1.1em;'>
        Enter your job description or requirements to get recommended SHL assessments.
    </p>
    """, unsafe_allow_html=True)

    with st.form("recommendation_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_area(
                "Enter job description or requirements:",
                placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams...",
                height=150
            )
        with col2:
            max_duration = st.number_input(
                "Maximum duration (minutes):",
                min_value=0,
                value=60,
                help="Set to 0 for no duration limit"
            )
            similarity_threshold = st.slider(
                "Similarity Threshold:",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Higher values mean more strict matching"
            )
        submit = st.form_submit_button("Get Recommendations")

    if submit and query:
        with st.spinner("🔎 Getting recommendations..."):
            try:
                recommender = SHLRecommender()
                if recommender is None:
                    st.error("Failed to initialize the recommender. Please try again later.")
                    return
                    
                recs = recommender.recommend(
                    query, 
                    max_duration if max_duration > 0 else None,
                    similarity_threshold=similarity_threshold
                )
                
                st.markdown("---")
                if not recs:
                    st.warning("No recommendations found. Try lowering the similarity threshold or using more general keywords.")
                else:
                    st.success(f"Found {len(recs)} recommendations:")
                    
                    # Create a DataFrame with sorted recommendations
                    df = pd.DataFrame(recs)
                    
                    # Format the similarity score as percentage
                    df['similarity_score'] = df['similarity_score'].apply(lambda x: f"{x*100:.1f}%")
                    
                    # Reorder columns for better display
                    columns_order = [
                        'assessment_name',
                        'similarity_score',
                        'test_type',
                        'duration',
                        'remote_testing_support',
                        'adaptive_irt_support',
                        'assessment_url'
                    ]
                    df = df[columns_order]
                    
                    # Rename columns for better display
                    df = df.rename(columns={
                        'assessment_name': 'Assessment Name',
                        'similarity_score': 'Match Score',
                        'test_type': 'Test Type',
                        'duration': 'Duration',
                        'remote_testing_support': 'Remote Testing',
                        'adaptive_irt_support': 'Adaptive IRT',
                        'assessment_url': 'URL'
                    })
                    
                    # Display the DataFrame with custom styling
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Show detailed view
                    with st.expander("Show detailed recommendations"):
                        for idx, rec in enumerate(recs, 1):
                            st.markdown(f"### Recommendation {idx}")
                            st.markdown(f"**Assessment Name:** {rec['assessment_name']}")
                            st.markdown(f"**Match Score:** {float(rec['similarity_score'])*100:.1f}%")
                            st.markdown(f"**Test Type:** {rec['test_type']}")
                            st.markdown(f"**Duration:** {rec['duration']}")
                            st.markdown(f"**Remote Testing Support:** {'✅' if rec['remote_testing_support'] else '❌'}")
                            st.markdown(f"**Adaptive IRT Support:** {'✅' if rec['adaptive_irt_support'] else '❌'}")
                            st.markdown(f"**URL:** [{rec['assessment_url']}]({rec['assessment_url']})")
                            st.markdown("---")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in recommendation: {str(e)}", exc_info=True)

# FastAPI setup
app = FastAPI(title="SHL Assessment Recommender API")

class RecommendationRequest(BaseModel):
    query: str
    max_duration: Optional[int] = None
    similarity_threshold: Optional[float] = 0.3
    max_results: Optional[int] = 10

class Assessment(BaseModel):
    assessment_name: str
    assessment_url: str
    remote_testing_support: bool
    adaptive_irt_support: bool
    duration: str
    test_type: str
    similarity_score: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=List[Assessment])
def recommend(request: RecommendationRequest):
    try:
        recommender = SHLRecommender()
        results = recommender.recommend(
            query=request.query,
            max_duration=request.max_duration,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        return results
    except Exception as e:
        logger.error(f"Error in API recommendation: {str(e)}", exc_info=True)
        raise

# Add evaluation button to Streamlit sidebar
if 'st' in globals():
    with st.sidebar:
        if st.button('Run Evaluation Pipeline'):
            st.info('Running evaluation...')
            recommender = SHLRecommender()
            if recommender:
                mean_recall, map_k = evaluate_recommender(recommender, test_set, k=3)
                st.success(f"Mean Recall@3: {mean_recall:.3f}\nMAP@3: {map_k:.3f}")
            else:
                st.error('Recommender not initialized.')

if __name__ == "__main__":
    main() 
