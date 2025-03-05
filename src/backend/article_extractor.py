import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re

def extract_article_newspaper3k(url):
    """Extracts article content using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "error": None,
        }
    except Exception as e:
        return {"title": None, "text": None, "authors": None, "publish_date": None, "error": str(e)}

def extract_article_beautifulsoup(url):
    """Extracts article content using requests and BeautifulSoup."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        article_body = soup.find('div', class_='article__content') #Modify this based on the website
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = "\n".join([p.get_text(strip=True) for p in paragraphs])

            title_element = soup.find('h1') #or soup.title
            title = title_element.get_text(strip=True) if title_element else "No Title Found"

            return {
                "title": title,
                "text": article_text,
                "authors": [],  # Extracting authors with BS4 can be tricky
                "publish_date": None,  # Extracting date with BS4 can be tricky
                "error": None,
            }
        else:
             return {"title": None, "text": None, "authors": None, "publish_date": None, "error": "Could not find article content using BeautifulSoup"}

    except requests.exceptions.RequestException as e:
        return {"title": None, "text": None, "authors": None, "publish_date": None, "error": f"Request Error: {e}"}
    except Exception as e:
        return {"title": None, "text": None, "authors": None, "publish_date": None, "error": f"Error processing the page: {e}"}


def clean_text(text):
    """Removes unwanted characters and extra whitespace from the text."""
    if text is None:
        return None
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_article(url, use_newspaper3k=True):
    """Extracts article content from a URL, with a fallback to BeautifulSoup."""

    if use_newspaper3k:
        result = extract_article_newspaper3k(url)
        if result["error"] is None:  # newspaper3k worked!
             result["text"] = clean_text(result["text"])
             return result
        # else, fall through to BeautifulSoup

    result = extract_article_beautifulsoup(url)
    result["text"] = clean_text(result["text"])
    return result


# Removed the if __name__ == '__main__': block so it doesn't run on import.