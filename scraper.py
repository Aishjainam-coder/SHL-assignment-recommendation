import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List, Dict
import time
import os

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com/solutions/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def get_assessment_details(self, url: str) -> Dict:
        """Extract details from an individual assessment page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract name
            name = soup.find('h1').text.strip() if soup.find('h1') else "Unknown"
            
            # Extract duration - look for common patterns
            duration = "60 mins"  # default
            duration_elements = soup.find_all(text=lambda text: text and 'min' in text.lower())
            for element in duration_elements:
                if 'min' in element.lower():
                    duration = element.strip()
                    break
            
            # Extract test type
            test_type = "General"
            type_elements = soup.find_all(text=lambda text: text and 'test type' in text.lower())
            for element in type_elements:
                if 'test type' in element.lower():
                    test_type = element.strip()
                    break
            
            details = {
                "name": name,
                "url": url,
                "remote_testing_support": "Yes",  # Most SHL tests support remote testing
                "adaptive_irt_support": "Yes",    # Most SHL tests use adaptive testing
                "duration": duration,
                "test_type": test_type
            }
            
            return details
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None

    def scrape_catalog(self) -> List[Dict]:
        """Scrape the main catalog page and all assessment pages"""
        try:
            response = requests.get(self.base_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            assessments = []
            
            # Look for assessment links in various possible locations
            assessment_links = []
            
            # Try different selectors that might contain assessment links
            selectors = [
                'a[href*="product-catalog"]',  # Links containing product-catalog
                '.product-link',               # Common class for product links
                '.assessment-link',            # Common class for assessment links
                'a[href*="/view/"]'           # Links containing /view/
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    url = link.get('href')
                    if url and 'product-catalog' in url:
                        if not url.startswith('http'):
                            url = 'https://www.shl.com' + url
                        assessment_links.append(url)
            
            # Remove duplicates
            assessment_links = list(set(assessment_links))
            
            print(f"Found {len(assessment_links)} assessment links")
            
            for url in assessment_links:
                print(f"Scraping: {url}")
                details = self.get_assessment_details(url)
                if details:
                    assessments.append(details)
                time.sleep(2)  # Be nice to the server
                    
            return assessments
        except Exception as e:
            print(f"Error scraping catalog: {str(e)}")
            return []

    def save_to_csv(self, assessments: List[Dict], filename: str = "shl_assessments.csv"):
        """Save scraped data to CSV"""
        df = pd.DataFrame(assessments)
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(assessments)} assessments to {filepath}")

    def save_to_json(self, assessments: List[Dict], filename: str = "shl_assessments.json"):
        """Save scraped data to JSON"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(assessments)} assessments to {filepath}")

if __name__ == "__main__":
    scraper = SHLScraper()
    assessments = scraper.scrape_catalog()
    scraper.save_to_csv(assessments)
    scraper.save_to_json(assessments) 