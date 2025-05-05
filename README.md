# SHL Assessment Recommendation System

An intelligent recommendation system that suggests relevant SHL assessments based on job descriptions or natural language queries.

## Features

- Natural language processing for understanding job requirements
- Semantic search for finding relevant assessments
- Duration-based filtering
- Web interface for easy interaction
- REST API for integration
- Evaluation metrics (Mean Recall@K and MAP@K)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd shl-assessment-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the scraper to collect assessment data:
```bash
python scraper.py
```

4. Start the API server:
```bash
python api.py
```

5. Start the Streamlit frontend:
```bash
streamlit run app.py
```

## Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8501`
2. Enter your job description or requirements
3. Optionally set a maximum duration
4. Click "Get Recommendations"

### API
The API provides two endpoints:

1. Health Check:
```bash
GET http://localhost:8000/health
```

2. Get Recommendations:
```bash
POST http://localhost:8000/recommend
Content-Type: application/json

{
    "query": "I am hiring for Java developers...",
    "max_duration": 60
}
```

## Evaluation

The system can be evaluated using the provided test queries. Run the evaluation using:
```python
from recommender import SHLRecommender

recommender = SHLRecommender()
results = recommender.evaluate(test_queries)
print(results)
```

## Architecture

- **Data Collection**: Web scraper to collect SHL assessment data
- **Recommendation Engine**: Sentence transformers for semantic search
- **API**: FastAPI backend
- **Frontend**: Streamlit web interface

## Dependencies

- Python 3.9+
- FastAPI
- Streamlit
- sentence-transformers
- pandas
- numpy
- scikit-learn
- requests
- beautifulsoup4

## License

MIT License 