import os
import json
import argparse
import requests
import datetime
import time
import re
import traceback
import random

from tqdm import tqdm
from pathlib import Path

from fake_useragent import UserAgent
ua = UserAgent()

def get_transcription(talk_name, language):
    headers = {
        'user-agent': ua.random,
    }

    try:
        url =  "https://www.ted.com/talks/{}/transcript.json?language={}".format(talk_name, language)
        r = requests.get(url, headers=headers)
    except Exception as e:
        print('error', e)
        print('Retry:')
        time.sleep(10)
        headers = {
            'user-agent': ua.random,
        }
        url =  "https://www.ted.com/talks/{}/transcript.json?language={}".format(talk_name, language)
        r = requests.get(url, headers=headers)
        # traceback.print_exc()

    return r.json()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')


    parser.add_argument("talk_names_path", type=str)

    args = parser.parse_args()

    talk_names_path = args.talk_names_path

    with open(talk_names_path, "r", encoding="utf-8") as f:
        talks = json.load(f)

    count = 0
    talks_with_transcript = []
    for index, talk in tqdm(enumerate(talks), total=len(talks)):

        talk_name = talk['talk_name']
        for language in ['th', 'en']:
            data = get_transcription(talk_name, language)

            sentences = []
            if not 'paragraphs' in data.keys():
                print("There is no {} language for this talk :{}".format(language, talk_name))
                break
            count += 1
            cues = data['paragraphs']
            for cue in cues:
                for time_step in cue['cues']:
                    text = time_step['text']
                    text = re.sub(r"\s\n", " ", text) 
                    text = re.sub(r"\n", " ", text) 
                    sentences.append(text)

            talks[index]['transcript_{}'.format(language)] = sentences
            talks_with_transcript.append(talks)

            if index % 100 ==0 and index != 0:
                time.sleep(10)
    
    print("Total number of parallel talks : ", count)

    current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")

    with open("./data/dump_talks_th-en_transcript.{}.json".format(current_time), "w", encoding="utf-8") as f:

        talks = json.dump(talks, f, ensure_ascii=False, indent=4)


