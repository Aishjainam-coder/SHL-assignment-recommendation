from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from recommender import SHLRecommender
import uvicorn

app = FastAPI(title="SHL Assessment Recommender API")

# Initialize recommender
recommender = SHLRecommender()

class RecommendationRequest(BaseModel):
    query: str
    max_duration: Optional[int] = None

class RecommendationResponse(BaseModel):
    recommendations: List[dict]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Set a very high similarity threshold for strict filtering
        similarity_threshold = 0.99
        # Require at least 3 words in the query
        if not request.query or len(request.query.strip().split()) < 3:
            return {"recommendations": []}
        recommendations = recommender.recommend(
            query=request.query,
            max_duration=request.max_duration,
            similarity_threshold=similarity_threshold
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 