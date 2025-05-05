import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import json
import os

class SHLRecommender:
    def __init__(self, data_path: str = "data/shl_assessments.json"):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.load_data(data_path)
        self.prepare_embeddings()

    def load_data(self, data_path: str):
        """Load assessment data from JSON file"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.assessments = json.load(f)
        except FileNotFoundError:
            print(f"Data file {data_path} not found. Using empty list.")
            self.assessments = []

    def prepare_embeddings(self):
        """Create embeddings for all assessments"""
        assessment_texts = [
            f"{assess['name']} {assess['test_type']} {assess['duration']}"
            for assess in self.assessments
        ]
        self.embeddings = self.model.encode(assessment_texts)

    def extract_duration(self, text: str) -> Optional[int]:
        """Extract duration in minutes from text"""
        import re
        duration_pattern = r'(\d+)\s*(?:min|minute|mins|minutes)'
        match = re.search(duration_pattern, text.lower())
        if match:
            return int(match.group(1))
        return None

    def recommend(self, query: str, max_duration: Optional[int] = None, max_results: int = 10, similarity_threshold: float = 0.5) -> List[Dict]:
        """Get recommendations based on query, with a similarity threshold to filter out irrelevant results"""
        # Encode query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if len(results) >= max_results:
                break
            if similarities[idx] < similarity_threshold:
                continue
            assessment = self.assessments[idx]
            # Filter by duration if specified
            if max_duration:
                duration = self.extract_duration(assessment['duration'])
                if duration and duration > max_duration:
                    continue
            results.append({
                "assessment_name": assessment['name'],
                "assessment_url": assessment['url'],
                "remote_testing_support": assessment['remote_testing_support'],
                "adaptive_irt_support": assessment['adaptive_irt_support'],
                "duration": assessment['duration'],
                "test_type": assessment['test_type']
            })
        return results

    def evaluate(self, test_queries: List[Dict], k: int = 3) -> Dict:
        """Evaluate the recommender using Mean Recall@K and MAP@K"""
        recalls = []
        aps = []
        
        for query_data in test_queries:
            query = query_data['query']
            relevant_assessments = query_data['relevant_assessments']
            
            # Get recommendations
            recommendations = self.recommend(query, max_results=k)
            recommended_urls = [r['assessment_url'] for r in recommendations]
            
            # Calculate Recall@K
            relevant_found = sum(1 for url in relevant_assessments if url in recommended_urls)
            recall = relevant_found / len(relevant_assessments) if relevant_assessments else 0
            recalls.append(recall)
            
            # Calculate AP@K
            ap = 0
            relevant_count = 0
            for i, url in enumerate(recommended_urls, 1):
                if url in relevant_assessments:
                    relevant_count += 1
                    ap += relevant_count / i
            ap = ap / min(k, len(relevant_assessments)) if relevant_assessments else 0
            aps.append(ap)
        
        return {
            "mean_recall@k": np.mean(recalls),
            "map@k": np.mean(aps)
        } 