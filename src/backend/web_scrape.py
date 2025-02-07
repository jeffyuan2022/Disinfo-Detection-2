import pandas as pd
import re
import time
import requests
from bs4 import BeautifulSoup

# Function to scrape one page of fact-checks
def scrape_fact_checks(page_number):
    url = f"https://www.politifact.com/factchecks/?page={page_number}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # List to store the fact-check data
    data = []

    # Find all fact-check articles on the page
    fact_checks = soup.find_all('article', class_='m-statement')

    for fact in fact_checks:
        # Extract Author/Speaker
        author = fact.find('a', class_='m-statement__name').text.strip()

        # Extract the Date of the statement
        date_string = fact.find('div', class_='m-statement__desc').text.strip()

        # Use a regular expression to extract only the date portion (e.g., October 8, 2024)
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4})', date_string)
        date = date_match.group(0) if date_match else "No date found"

        # Extract the Claim (statement being fact-checked)
        claim = fact.find('div', class_='m-statement__quote').find('a').text.strip()

        # Extract the URL to the full fact-check article
        link = "https://www.politifact.com" + fact.find('div', class_='m-statement__quote').find('a')['href']

        # Extract the Rating (e.g., False, Pants on Fire)
        rating = fact.find('div', class_='m-statement__meter').find('img')['alt'].strip()

        # Append the extracted information to the list
        data.append({
            'Author/Speaker': author,
            'Date': date,
            'Claim': claim,
            'Rating': rating,
            'Link to Full Article': link
        })

    return data

# Loop through multiple pages and collect data
def scrape_multiple_pages(start_page, end_page):
    all_data = []
    for page_number in range(start_page, end_page + 1):
        print(f"Scraping page {page_number}...")
        page_data = scrape_fact_checks(page_number)
        all_data.extend(page_data)
        time.sleep(2)  # Sleep for 2 seconds between each page request

    return all_data

# Scrape data from page 1 to 1
data = scrape_multiple_pages(1, 1)
politifact_data = pd.DataFrame(data)

def scrape_fact_check_details(url):
    """
    Scrapes detailed information from a given PolitiFact fact-check URL
    and handles character encoding issues.
    """
    response = requests.get(url)
    
    # Decode content to UTF-8 to handle special characters
    content = response.content.decode('utf-8', errors='replace')
    soup = BeautifulSoup(content, 'html.parser')

    # Initialize a dictionary to store the scraped details
    details = {'URL': url}

    # Extract the quote
    quote_div = soup.find('div', class_='m-statement__quote')
    details['Quote'] = quote_div.text.strip() if quote_div else "Quote not found"

    # Extract the subline
    subline = soup.find('h1', class_='c-title c-title--subline')
    details['Subline'] = subline.text.strip() if subline else "Subline not found"

    # Extract the "If Your Time is Short" explanation
    callout_div = soup.find('div', class_='m-callout m-callout--large')
    if callout_div:
        body_div = callout_div.find('div', class_='m-callout__body')
        details['Explanation'] = body_div.get_text(separator="\n", strip=True) if body_div else "Explanation not found"
    else:
        details['Explanation'] = "Callout section not found"

    return details

# Scrape all detailed information for each link in the 'Link to Full Article' column
def scrape_all_links(dataframe):
    """
    Loops through all the links in the DataFrame's 'Link to Full Article' column 
    and scrapes detailed information for each link.
    """
    detailed_data = []
    for index, link in enumerate(dataframe['Link to Full Article']):
        print(f"Scraping details from link {index + 1}/{len(dataframe)}: {link}")
        try:
            details = scrape_fact_check_details(link)
            detailed_data.append(details)
        except Exception as e:
            print(f"Error scraping {link}: {e}")
            detailed_data.append({'URL': link, 'Quote': None, 'Subline': None, 'Explanation': None})
        time.sleep(2)  # Sleep for 2 seconds to avoid overloading the server
    return detailed_data

# Scrape details for all fact-check URLs
detailed_fact_checks = scrape_all_links(politifact_data)

# Convert the detailed data into a DataFrame
detailed_fact_checks_df = pd.DataFrame(detailed_fact_checks)

# Save the detailed data to a CSV file
detailed_fact_checks_df.to_csv('src/data/processed/detailed_fact_checks.csv', index=False, encoding="utf-8")
print("Scraping completed and saved to 'detailed_fact_checks.csv'.")