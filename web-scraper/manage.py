from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import requests
import json


class WebScraper:
    def __init__(self, url):
        self.url = url

    
    def scrape(self):
        response = requests.get(self.url)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch page. Status code: {response.status_code}")

        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    

    def get_text(self, soup):
        text = soup.get_text()
        return text
    

    def convert_text_to_vector(self, text):
        text = list(filter(lambda x: x != '', text.split('\n')))
        vectorizer = CountVectorizer()

        return vectorizer.fit_transform(text)


if __name__ == "__main__":
    with open('data/links.json', "r") as fileLinkList:
        urls = json.load(fileLinkList)

    with open('data/answers.json', "r") as fileAnswers:
        answers = json.load(fileAnswers)

    if len(urls) != len(answers):
        print("Error: The number of URLs and answers do not match.")
        exit(1)

    combined_data = []

    for url, answer in zip(urls, answers):
        scraper = WebScraper(url)

        try:
            soup = scraper.scrape()
            combined_data.append({
                "description": scraper.get_text(soup),
                "labels": answer
            })
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    # Save combined data to data.json
    with open('data/data.json', "w") as fileData:
        json.dump(combined_data, fileData, indent=4)

    print("Combined data has been saved to data/data.json")