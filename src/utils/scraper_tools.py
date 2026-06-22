import requests
from bs4 import BeautifulSoup

def scrape_text_from_url(url: str) -> str:
    """Smartly scrapes text content, extracting headings and paragraphs while filtering out noise."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Clean up DOM: Remove scripts, styles, and structural/navigation tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.extract()
            
        # 2. Extract tags that hold the core content
        target_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
        extracted_elements = soup.find_all(target_tags)
        
        # 3. Filter empty text and reconstruct the document
        text_blocks = []
        for elem in extracted_elements:
            text = elem.get_text(strip=True)
            # Only keep elements with actual content (length > 20) or important headings
            if len(text) > 20 or elem.name in ['h1', 'h2', 'h3']: 
                text_blocks.append(text)
                
        full_text = "\n".join(text_blocks)
    
        return full_text
        
    except Exception as e:
        return f"Error scraping data from {url}: {e}"