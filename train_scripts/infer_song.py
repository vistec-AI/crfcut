import pycrfsuite
from pythainlp.tokenize import word_tokenize
import argparse

_CRFCUT_DATA_FILENAME = "models/song-crf.model"
_tagger = pycrfsuite.Tagger()
_tagger.open(_CRFCUT_DATA_FILENAME)

starters = [
    [
        "ไม่",
        "เธอ",
        "ฉัน",
        "ก็",
        "แต่",
        "จะ",
        "อยาก",
        "ให้",
        "แค่",
        "ถ้า",
        "ที่",
        "และ",
        "มัน",
        "อย่า",
        "เพราะ",
        "รัก",
        "ว่า",
        "คน",
        "เมื่อ",
        "มี",
        "อยู่",
        "ได้",
        "ขอ",
        "เป็น",
        "ใน",
        "หาก",
        "ต้อง",
        "บอก",
        "ยัง",
        "แล้ว",
        "แม้",
        "I",
        "รู้",
        "ใจ",
        "ไม่ต้อง",
        "เรา",
        "ถึง",
        "มา",
        "เหมือน",
        "ยิ่ง",
        "ใคร",
        "คง",
        "บ่",
        "อ้าย",
        "สิ่ง",
        "จาก",
        "เจ็บ",
        "ช่วย",
        "หรือ",
        "กับ",
    ]
]

enders = [
    "เธอ",
    "ไป",
    "กัน",
    "ไหม",
    "ฉัน",
    "หัวใจ",
    "ใคร",
    "ไหน",
    "ใจ",
    "เลย",
    "มา",
    "ดี",
    "เข้าใจ",
    "ได้",
    "ๆ",
    "อะไร",
    "นี้",
    "เท่าไร",
    "อยู่",
    "เจอ",
    "ไว้",
    "รัก",
    "แล้ว",
    "รู้",
    "เรา",
    "นั้น",
    "ไกล",
    "เขา",
    "เสียใจ",
    "มี",
    "ไหว",
    "พอ",
    "หรือเปล่า",
    "สักที",
    "ใหม่",
    "ยังไง",
    "คน",
    "ใช่ไหม",
    "ตาย",
    "เท่านั้น",
    "อย่างนี้",
    "นะ",
    "ที",
    "อย่างไร",
    "มากมาย",
    "เก่า",
    "น้ำตา",
    "นาน",
    "ทุกวัน",
    "คนเดียว",
]


def extract_features(doc, window=3, max_n_gram=3):
    #     #paddings for POS
    #     doc_pos = ['xxpad' for i in range(window)] + \
    #         [p for (w,p) in pos_tag(doc,engine='perceptorn', corpus='orchid')] + ['xxpad' for i in range(window)]

    # padding for words
    doc = (
        ["xxpad" for i in range(window)]
        + doc
        + ["xxpad" for i in range(window)]
    )

    doc_ender = []
    doc_starter = []
    # add enders
    for i in range(len(doc)):
        if doc[i] in enders:
            doc_ender.append("ender")
        else:
            doc_ender.append("normal")
    # add starters
    for i in range(len(doc)):
        if doc[i] in starters:
            doc_starter.append("starter")
        else:
            doc_starter.append("normal")

    doc_features = []
    # for each word
    for i in range(window, len(doc) - window):
        # bias term
        word_features = ["bias"]

        # ngram features
        for n_gram in range(1, min(max_n_gram + 1, 2 + window * 2)):
            for j in range(i - window, i + window + 2 - n_gram):
                feature_position = f"{n_gram}_{j-i}_{j-i+n_gram}"

                # word
                word_ = f'{"|".join(doc[j:(j+n_gram)])}'
                word_features += [f"word_{feature_position}={word_}"]

                #                #pos
                #                 pos_ =f'{"|".join(doc_pos[j:(j+n_gram)])}'
                #                 word_features += [f'pos_{feature_position}={pos_}']

                # enders
                ender_ = f'{"|".join(doc_ender[j:(j+n_gram)])}'
                word_features += [f"ender_{feature_position}={ender_}"]

                # starters
                starter_ = f'{"|".join(doc_starter[j:(j+n_gram)])}'
                word_features += [f"starter_{feature_position}={starter_}"]

        # number of tokens to the left and right
        nb_left = 0
        for l in range(i)[::-1]:
            if doc[l] == "<space>":
                break
            if True:
                nb_left += 1.0
        nb_right = 0
        for r in range(i + 1, len(doc)):
            if doc[r] == "<space>":
                break
            if True:
                nb_right += 1.0
        word_features += [f"nb_left={nb_left}", f"nb_right={nb_right}"]

        # append to feature per word
        doc_features.append(word_features)
    return doc_features


def segment(text: str):
    """
    CRF-based sentence segmentation.
    :param str text: text to be tokenized to sentences
    :return: list of words, tokenized from the text
    """
    toks = word_tokenize(text)
    feat = extract_features(toks)
    labs = _tagger.tag(feat)
    labs[-1] = "E"  # make sure it cuts the last sentence

    sentences = []
    sentence = ""
    for i, w in enumerate(toks):
        sentence = sentence + w
        if labs[i] == "E":
            sentences.append(sentence)
            sentence = ""

    return sentences


def main():
    parser = argparse.ArgumentParser(description="Segment songs")
    parser.add_argument(
        "--text",
        type=str,
        default="ชีวิตมันต้องเดินตามหาความฝัน หกล้มคลุกคลานเท่าไหร่ มันจะไปจบที่ตรงไหน เมื่อเดินเท่าไหร่มันก็ไปไม่ถึง",
    )
    args = parser.parse_args()
    res = segment(args.text)
    print(res)


if __name__ == "__main__":
    main()
