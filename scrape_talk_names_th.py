

"""
This script is to scrape the names TED talks from https://www.ted.com/talks that are available in Thai language only.

"""

import os
import time
import json
import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

LAST_PAGE_NUMBER = 44
URL = "https://www.ted.com/talks?language=th"

def loop_pages(preifx_url, last_page_number=LAST_PAGE_NUMBER):
    all_talks = []

    for page_number in range(1, last_page_number + 1, 1):
        print('current page number ', page_number)

        talks = []
        url = "{}&page={}".format(preifx_url, page_number)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        items = soup.find_all("div", {"class": "media__message"})#.find("a", {"data-ga-context": "talks"})

        talks = []
        for item in items:
            # print(item)
            # print("type(item)", type(item))
            talk_name = item.find("a", {"data-ga-context": "talks"}).attrs['href']
            search_object = re.search('(\/talks\/)(.*)(\?language=th)', talk_name)
            talk_name = search_object.group(2)

            talk_title = item.text
            # print(talk_name)
            obj = {
                "talk_name": talk_name,
                "talk_title": talk_title

            }
            talks.append(obj)
        
        print('number of talks for this page ', len(talks))

        all_talks.extend(talks)

        print('Accumulated number of talks ', len(all_talks))
        time.sleep(1)
    return all_talks

if __name__ == '__main__':

    data = loop_pages(URL, LAST_PAGE_NUMBER)
    print('length:', len(data))
    # print(data)
    with open('dump_talks_with_thai_transcript.{}.json'.format(), 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
