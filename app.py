import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
import json
import os
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# --- SHLRecommender class (with improved error handling) ---
class SHLRecommender:
    def __init__(self, data_path: str = "data/shl_assessments.json"):
        try:
            logger.info("Initializing SHLRecommender...")
            model_path = os.path.join("models", "all-MiniLM-L6-v2")
            if os.path.exists(model_path):
                self.model = SentenceTransformer(model_path)
                logger.info("Loaded model from local folder.")
            else:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("Loaded model from HuggingFace hub.")
            self.load_data(data_path)
            self.prepare_embeddings()
            logger.info("SHLRecommender initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHLRecommender: {str(e)}")
            st.error(f"Error initializing the recommender: {str(e)}")
            raise

    def load_data(self, data_path: str):
        try:
            logger.info(f"Loading data from {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
            
            if not isinstance(self.assessments, list):
                raise ValueError("Invalid data format: expected a list of assessments")
            
            logger.info(f"Successfully loaded {len(self.assessments)} assessments")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            st.error("Error parsing the data file. Please check the file format.")
            self.assessments = []
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            self.assessments = []

    def prepare_embeddings(self):
        try:
            logger.info("Preparing embeddings...")
            if not self.assessments:
                logger.warning("No assessments available for embedding")
                return
                
            assessment_texts = [
                f"{assess['name']} {assess['test_type']} {assess['duration']}"
                for assess in self.assessments
            ]
            self.embeddings = self.model.encode(assessment_texts)
            logger.info(f"Successfully prepared embeddings for {len(assessment_texts)} assessments")
        except Exception as e:
            logger.error(f"Error preparing embeddings: {str(e)}")
            st.error(f"Error preparing embeddings: {str(e)}")
            self.embeddings = None

    def extract_duration(self, text: str) -> Optional[int]:
        import re
        duration_pattern = r'(\d+)\s*(?:min|minute|mins|minutes)'
        match = re.search(duration_pattern, text.lower())
        if match:
            return int(match.group(1))
        return None

    def recommend(
        self, 
        query: str, 
        max_duration: Optional[int] = None, 
        max_results: int = 10, 
        similarity_threshold: float = 0.1  # Lowered threshold for better matching
    ) -> List[Dict]:
        try:
            if self.embeddings is None or len(self.embeddings) == 0:
                logger.error("No embeddings available for recommendation")
                return []
            
            # Preprocess query
            query = query.lower().strip()
            
            # Get query embedding
            query_embedding = self.model.encode(query)
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Get top indices
            top_indices = np.argsort(similarities)[::-1]
            
            # Debug logging
            logger.info(f"Query: {query}")
            logger.info(f"Top 5 similarities: {similarities[top_indices[:5]]}")
            
            results = []
            
            for idx in top_indices:
                if len(results) >= max_results:
                    break
                    
                assessment = self.assessments[idx]
                
                # Check duration if specified
                if max_duration:
                    duration = self.extract_duration(assessment['duration'])
                    if duration and duration > max_duration:
                        continue
                
                # Add to results if similarity meets threshold
                if similarities[idx] >= similarity_threshold:
                    results.append({
                        "assessment_name": assessment['name'],
                        "assessment_url": assessment['url'],
                        "remote_testing_support": assessment['remote_testing_support'],
                        "adaptive_irt_support": assessment['adaptive_irt_support'],
                        "duration": assessment['duration'],
                        "test_type": assessment['test_type'],
                        "similarity_score": float(similarities[idx])  # Add similarity score for debugging
                    })
            
            logger.info(f"Results found: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            st.error(f"Error getting recommendations: {str(e)}")
            return []

def verify_data_file(data_path: str) -> bool:
    """Verify that the data file exists and is valid."""
    try:
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            return False
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error("Invalid data format: expected a list of assessments")
            return False
            
        if not data:
            logger.error("Data file is empty")
            return False
            
        # Verify required fields in the first assessment
        if data and not all(key in data[0] for key in ['name', 'test_type', 'duration']):
            logger.error("Data file missing required fields")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error verifying data file: {str(e)}")
        return False

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
                    
                recommendations = recommender.recommend(
                    query, 
                    max_duration if max_duration > 0 else None
                )
                
                st.markdown("---")
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations:")
                    
                    # Create a DataFrame with sorted recommendations
                    df = pd.DataFrame(recommendations)
                    df = df.sort_values('similarity_score', ascending=False)
                    
                    # Display the DataFrame
                    st.dataframe(df, use_container_width=True)
                    
                    # Show detailed view
                    with st.expander("Show detailed recommendations"):
                        for idx, rec in enumerate(recommendations, 1):
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
                else:
                    st.warning("""
                    No recommendations found. Try:
                    1. Using more specific job requirements
                    2. Increasing the maximum duration
                    3. Using different keywords
                    """)
            except Exception as e:
                logger.error(f"Error getting recommendations: {str(e)}")
                st.error(f"An error occurred while getting recommendations: {str(e)}")

if __name__ == "__main__":
    main() 
