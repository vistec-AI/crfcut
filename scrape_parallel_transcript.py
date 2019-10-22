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
        traceback.print_exc()
        print("Try again.")
        time.sleep(10)
        headers = {
            'user-agent': ua.random,
        }
        url =  "https://www.ted.com/talks/{}/transcript.json?language={}".format(talk_name, language)
        r = requests.get(url, headers=headers)

    return r.json()


def extract_sentences(transcript):
    sentences = []
    cues = transcript['paragraphs']
    for cue in cues:
        for time_step in cue['cues']:
            text = time_step['text']
            text = re.sub(r"\s\n", " ", text) 
            text = re.sub(r"\n", " ", text) 
            sentences.append(text)
    return sentences

def scrape(talks):
    count = 0

    talks_with_transcript = []
    for index, talk in tqdm(enumerate(talks), total=len(talks)):

        talk_name = talk['talk_name']
        transcript_th = get_transcription(talk_name, 'th')
        transcript_en = get_transcription(talk_name, 'en')

        sentences = []
        if not 'paragraphs' in transcript_th.keys() or not 'paragraphs' in transcript_en.keys() :
            print("There is no th-en pair for this talk : `{}`".format(talk_name))
            continue

        talks[index]['transcript_th'] = extract_sentences(transcript_th)
        talks[index]['transcript_en'] = extract_sentences(transcript_en)

        talks_with_transcript.append(talks)
        count += 1

        if index % 200 == 0 and index != 0:
            time.sleep(random.choice([3, 5, 7, 10]))

    print("Total number of parallel talks : ", count)

    return talks_with_transcript

def save(talks_with_transcript):
    

    current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M")
    output_path = "./data/talks.transcript.th-en.{}.json".format(current_time)
    print('Writing file to {}'.format(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(talks_with_transcript, f, ensure_ascii=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')


    parser.add_argument("talk_names_path", type=str)

    args = parser.parse_args()

    talk_names_path = args.talk_names_path

    with open(talk_names_path, "r", encoding="utf-8") as f:
        talks = json.load(f)

    scaraped_talk = scrape(talks)
    save(scaraped_talk)