import json
from recommender import SHLRecommender
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt

def load_test_queries(filename: str = "test_queries.json") -> List[Dict]:
    """Load test queries from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation(ks: List[int] = [1, 3, 5, 10]) -> Dict:
    """Run evaluation for different K values"""
    recommender = SHLRecommender()
    test_queries = load_test_queries()
    
    results = {}
    for k in ks:
        metrics = recommender.evaluate(test_queries, k=k)
        results[f"k={k}"] = metrics
    
    return results

def plot_results(results: Dict):
    """Plot evaluation results"""
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['mean_recall@k'], marker='o', label='Mean Recall@K')
    plt.plot(df.index, df['map@k'], marker='o', label='MAP@K')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics vs K')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation_results.png')
    plt.close()

def main():
    print("Running evaluation...")
    results = run_evaluation()
    
    # Print results
    print("\nEvaluation Results:")
    for k, metrics in results.items():
        print(f"\n{k}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Plot results
    plot_results(results)
    print("\nResults plotted in evaluation_results.png")

if __name__ == "__main__":
    main() 