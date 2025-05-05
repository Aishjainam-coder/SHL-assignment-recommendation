import streamlit as st
import requests
import pandas as pd
from typing import List, Dict

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

# API configuration
API_URL = "http://localhost:8000"  # Change this to your deployed API URL

def get_recommendations(query: str, max_duration: int = None) -> List[Dict]:
    """Get recommendations from the API"""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"query": query, "max_duration": max_duration}
        )
        response.raise_for_status()
        return response.json()["recommendations"]
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
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
    <h1 style='text-align: center; color: #4F8BF9;'>SHL Assessment Recommender</h1>
    <p style='text-align: center; color: #333; font-size: 1.1em;'>
        Enter your job description or requirements to get recommended SHL assessments.
    </p>
    """, unsafe_allow_html=True)

    # Input form in a card-like container
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
            recommendations = get_recommendations(query, max_duration if max_duration > 0 else None)
            st.markdown("---")
            if recommendations:
                st.success(f"Found {len(recommendations)} recommendations:")
                st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                with st.expander("Show details for each recommendation"):
                    for rec in recommendations:
                        st.markdown(f"<b>Assessment Name:</b> {rec.get('assessment_name', '')}", unsafe_allow_html=True)
                        st.markdown(f"<a href='{rec.get('assessment_url', '')}' target='_blank'>Assessment URL</a>", unsafe_allow_html=True)
                        st.markdown(f"Remote Testing Support: <b>{rec.get('remote_testing_support', '')}</b>", unsafe_allow_html=True)
                        st.markdown(f"Adaptive IRT Support: <b>{rec.get('adaptive_irt_support', '')}</b>", unsafe_allow_html=True)
                        duration = rec.get('duration', '')
                        if duration and len(duration) < 50 and '<' not in duration and 'function' not in duration:
                            st.markdown(f"Duration: <b>{duration}</b>", unsafe_allow_html=True)
                        st.markdown(f"Test Type: <b>{rec.get('test_type', '')}</b>", unsafe_allow_html=True)
                        st.markdown('<hr style="margin: 0.5em 0;">', unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try adjusting your query or duration limit.")

if __name__ == "__main__":
    main() 