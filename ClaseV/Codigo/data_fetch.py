"""
Author: Abraham R.

Description: 
This module serves as a simple use case of extracting data from websites to feed into an LLM.

"""
import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):

    response = requests.get(url)
    

    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = "\n".join(paragraphs)
        return text.strip()
    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return None

# test
url = "https://en.wikipedia.org/wiki/University_of_Buenos_Aires"
text = extract_text_from_url(url)
print(text)
