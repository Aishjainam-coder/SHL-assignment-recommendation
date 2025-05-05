# SHL Assessment Recommendation System – Approach Document

## Problem Statement
Hiring managers face challenges in selecting suitable SHL assessments for various roles due to inefficient keyword-based search. The goal is to build an intelligent recommendation system that, given a natural language query or job description, returns the most relevant SHL assessments with key attributes.

---

## Solution Overview
We developed an end-to-end web application that:
- Accepts a natural language query or job description.
- Returns up to 10 relevant SHL assessments in a tabular format.
- Provides an accessible API and a user-friendly frontend.

---

## Key Components & Tools
- **Data Collection:**  
  Built a custom web scraper using `requests` and `BeautifulSoup` to extract assessment details (name, URL, duration, test type, remote/adaptive support) from the [SHL product catalog](https://www.shl.com/solutions/products/product-catalog/).  
  Data stored in CSV/JSON for efficient access.

- **Semantic Search & Recommendation Engine:**  
  Used `sentence-transformers` (SBERT) to encode both assessment descriptions and user queries into vector embeddings.  
  Employed cosine similarity to rank and retrieve the most relevant assessments for a given query.

- **API:**  
  Developed with `FastAPI` for high performance and easy documentation.  
  `/health` endpoint for health checks.  
  `/recommend` endpoint accepts a query and returns recommendations in the required JSON format.

- **Frontend:**  
  Built with `Streamlit` for rapid prototyping and accessibility.  
  Allows users to input queries, set duration limits, and view recommendations in a table and detailed format.

- **Evaluation:**  
  Used provided test queries and ground-truth data.  
  Computed **Mean Recall@3** and **MAP@3** to measure retrieval accuracy.  
  Iteratively improved the model and data cleaning to maximize these metrics.

---

## Libraries & Technologies
- Python 3.11+
- `requests`, `beautifulsoup4` (scraping)
- `pandas`, `numpy` (data handling)
- `sentence-transformers`, `scikit-learn` (semantic search)
- `FastAPI`, `uvicorn` (API)
- `Streamlit` (frontend)

---

## Optimization & Robustness
- Cleaned and filtered scraped data to avoid noise and irrelevant content.
- Limited recommendations to individual test solutions and filtered by duration if specified.
- Ensured all endpoints return proper HTTP status codes and JSON responses.
- Added error handling and user feedback in the frontend.

---

## Evaluation Results
- Achieved high Mean Recall@3 and MAP@3 on the provided test set.
- System returns relevant, well-ranked assessments for a variety of job queries.

---

## Accessibility
- API and frontend are accessible via public URLs.
- Codebase is available on GitHub with clear documentation and setup instructions.

---

**In summary, this system leverages modern NLP, robust data engineering, and accessible web technologies to deliver a practical, accurate, and user-friendly SHL assessment recommendation solution.** 