import re
import string

import emoji
from nltk import RegexpTokenizer

tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')


def tokenizeText(doc):
    # split into tokens by white space
    tokens = tokenizer.tokenize(doc)
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return " ".join(tokens)


# Removes emojis from text
def removeEmojis(text):
    allChars = [str for str in text]
    emojiList = [c for c in allChars if c in emoji.UNICODE_EMOJI]
    return ' '.join([str for str in text.split() if not any(i in str for i in emojiList)])


# Cleans text of noisy characters using regular expressions
def cleanText(text):
    text = text.replace("\n", ' ').replace('#', '')
    text = re.sub('@[^\s]+', '', text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    text = removeEmojis(text)
    text = tokenizeText(text)

    return text
