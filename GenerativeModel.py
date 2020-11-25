from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from torch.jit import script, trace
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
from io import open
import codecs
import itertools
import math
import pandas as pd


class DepressionCounselor:
    def __init__(self, trainFile='ConversationalData.csv'):
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.data = self.loadData(trainFile)

    def showData(self, filename, n):
        with open('Data' + os.sep + filename, encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines[:n]:
            print(line)

    def loadData(self, filename):
        conversationPair = []
        data = pd.read_csv('Data' + os.sep + filename, delimiter=',')
        data.fillna(value='', inplace=True)
        data = data.sample(frac=1)
        inputLines = data['Initial Text']
        targetLines = data['Response Text']
        for inputLine, targetLine in inputLines, targetLines:
            conversationPair.append([inputLine, targetLine])

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class VOC:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.numWords = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.numWords
            self.word2count[word] = 1
            self.index2word[self.numWords] = word
            self.numWords += 1
        else:
            self.word2count[word] += 1

    def trim(self, minimumCount):
        if self.trimmed:
            return

        self.trimmed = True
        keepWords = []

        for key, value in self.word2count.items():
            if value >= minimumCount:
                keepWords.append(key)

            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
            self.numWords = 3

            for word in keepWords:
                self.addWord(word)


if __name__ == '__main__':
    counselor = DepressionCounselor()
    counselor.loadData('ConversationalData.csv')
