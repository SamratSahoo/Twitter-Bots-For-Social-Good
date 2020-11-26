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

MAX_LENGTH = 30
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MIN_COUNT = 3


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


class DataPreprocessor:
    def __init__(self, trainFile='ConversationalData.csv'):
        self.trainFile = trainFile
        self.data = self.loadData(self.trainFile)
        self.voc, self.pairs = self.loadPrepareData(dataFile=self.trainFile,
                                                    corpusName=self.trainFile.replace('.csv', ''))
        self.trimmedPairs = self.trimRareWords(self.pairs)

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

        return conversationPair

    def unicodeToAscii(self, string):
        return ''.join(
            character for character in unicodedata.normalize('NFD', string)
            if unicodedata.category(character) != 'Mn'
        )

    def normalizeString(self, string):
        string = self.unicodeToAscii(string.lower().strip())
        string = re.sub(r"([.!?])", r"\1", string)
        string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
        string = re.sub(r"\s", r" ", string)
        return string

    def readVocs(self, filename, corpusName):
        pairs = []
        dataframe = pd.read_csv('Data' + os.sep + filename, delimiter=',')
        for index, row in dataframe.iterrows():
            pairs.append([
                self.normalizeString(row['Initial Text']),
                self.normalizeString(row['Response Text'])
            ])

        voc = VOC(corpusName)
        return voc, pairs

    def filterPair(self, pair):
        return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def loadPrepareData(self, corpusName, dataFile):
        voc, pairs = self.readVocs(dataFile, corpusName)
        pairs = self.filterPairs(pairs)
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])

        return voc, pairs

    def trimRareWords(self, pairs):
        self.voc.trim(MIN_COUNT)
        keepPairs = []
        for pair in pairs:
            inputSequence = []
            outputSequence = []
            keepInput = True
            keepOutput = True

            for word in inputSequence.split(' '):
                if word not in self.voc.word2index:
                    keepInput = False
                    break

            for word in outputSequence.split(' '):
                if word not in self.voc.word2index:
                    keepOutput = False
                    break

            if keepOutput and keepInput:
                keepPairs.append(pair)

        return keepPairs

    def indexesFromSentences(self, sentence):
        return [self.voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]

    def zeroPadding(self, list, fillValue=PAD_TOKEN):
        return list(itertools.zip_longest(*list, fillvalue=fillValue))

    def binaryMatrix(self, list, value=PAD_TOKEN):
        matrix = []
        for i, seq in enumerate(list):
            matrix.append([])
            for token in seq:
                if token == PAD_TOKEN:
                    matrix.append(0)
                else:
                    matrix.append(1)

        return matrix

    def inputVar(self, list):
        indexesBatch = [self.indexesFromSentences(self.voc, sentence) for sentence in list]
        lengths = torch.tensor([len(indexes) for indexes in indexesBatch])
        padList = self.zeroPadding(list)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def outputVar(self, list):
        indexesBatch = [self.indexesFromSentences(self.voc, sentence) for sentence in list]
        maxTagetLength = max([len(indexes) for indexes in indexesBatch])
        padList = self.zeroPadding(list)
        mask = self.binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, maxTagetLength

    def batch2TrainData(self, pairBatch):
        pairBatch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        inputBatch, outputBatch = [], []
        for pair in pairBatch:
            inputBatch.append(pair[0])
            outputBatch.append(pair[0])
        inp, lengths, = self.inputVar(inputBatch)
        output, mask, maxTargetLength = self.outputVar(outputBatch)
        return inp, lengths, output, mask, maxTargetLength


class EncoderRNN(nn.Module):
    def __init__(self, hiddenSize, embedding, nLayers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.nLayers = nLayers
        self.hiddenSize = hiddenSize
        self.embedding = embedding

        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=(0 if nLayers == 1 else dropout), bidirectional=True)

    def forward(self, inputSequence, inputLengths, hidden=None):
        embedded = self.embedding(inputSequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :, self.hiddenSize] + outputs[:, :, self.hiddenSize, :]
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hiddenSize, method='dot'):
        super(Attn, self).__init__()
        self.method = method
        self.hiddenSize = hiddenSize
        if self.method == 'general':
            self.attn = nn.Liner(self.hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hiddenSize * 2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(hiddenSize))

    def dotScore(self, hidden, encoderOutput):
        return torch.sum(hidden * encoderOutput, dim=2)

    def generalScore(self, hidden, encoderOutput):
        energy = self.attn(encoderOutput)
        return torch.sum(hidden * energy, dim=2)

    def concatScore(self, hidden, encoderOutput):
        energy = self.attn(torch.cat((hidden.expand(encoderOutput.size(0), -1, -1), encoderOutput), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoderOutputs):
        if self.method == 'general':
            attnEnergies = self.generalScore(hidden, encoderOutputs)
        elif self.method == 'concat':
            attnEnergies = self.concatScore(hidden, encoderOutputs)
        if self.method == 'dot':
            attnEnergies = self.dotScore(hidden, encoderOutputs)

        attnEnergies = attnEnergies.t()

        return F.softmax(attnEnergies, dim=1).unsqueeze(1)


class DecoderRNN(nn.Module):
    def __init__(self, attnModel, embedding, hiddenSize, outputSize, nLayers=1, dropout=0):
        super(DecoderRNN, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attnModel = attnModel
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.nLayers = nLayers
        self.dropout = dropout

        self.embedding = embedding
        self.embeddingDropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=(0 if nLayers == 1 else dropout))

        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.attn = Attn(attnModel, hiddenSize)

    def forward(self, inputStep, lastHidden, encoderOutputs):
        embedded = self.forward(inputStep)
        embedded = self.embeddingDropout(embedded)
        rnnOutput, hidden = self.gru(embedded, lastHidden)
        attnWeights = self.attn(rnnOutput, encoderOutputs)
        context = attnWeights.bmm(encoderOutputs.transpose(0, 1))
        rnnOutput = rnnOutput.squeeze(0)
        context = context.squeeze(1)
        concatInput = torch.cat((rnnOutput, context), 1)
        concatOutput = torch.tanh(self.concat(concatInput))
        output = self.out(concatOutput)
        output = F.softmax(output, dim=1)
        return output, hidden

    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()


class GenerativeModel:
    def __init__(self, encoder, decoder, encoderOptimizer, decoderOptimizer, batchSize, teacherForcingRatio=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = encoder
        self.decoder = decoder
        self.encoderOptimizer = encoderOptimizer
        self.decoderOptimizer = decoderOptimizer

        self.batchSize = batchSize
        self.teacherForcingRatio = teacherForcingRatio

    def train(self, inputVariable, targetVariable, mask, lengths, maxTargetLength, clip, embedding=None,
              maxLength=MAX_LENGTH):
        self.encoderOptimizer.zero_grad()
        self.decoderOptimizer.zero_grad()
        inputVariable = inputVariable.to(self.device)
        targetVariable = targetVariable.to(self.device)
        mask = mask.to(self.device)
        lengths = lengths.to('cpu')

        loss = 0
        printLosses = []
        nTotals = 0

        encoderOutputs, encoderHidden = self.encoder(inputVariable, lengths)

        decoderInput = torch.LongTensor([[SOS_TOKEN for _ in range(self.batchSize)]])
        decoderInput = decoderInput.to(self.device)
        decoderHidden = encoderHidden[:self.decoder.nLayers]

        useTeacherForcing = random.random() < self.teacherForcingRatio

        if useTeacherForcing:
            for t in range(maxTargetLength):
                decoderOutput, decoderHidden = self.decoder(
                    decoderInput, decoderHidden, encoderOutputs
                )

                decoderInput = targetVariable[t].view(1, -1)
                maskLoss, nTotal = self.decoder.maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
                loss += maskLoss
                printLosses.append(maskLoss.item() * nTotal)
                nTotals += nTotal

        else:
            for t in range(maxTargetLength):
                decoderOutput, decoderHidden = self.decoder(
                    decoderInput, decoderHidden, encoderOutputs
                )
                _, topi = decoderOutput.topk(1)
                decoderInput = torch.LongTensor([[topi[i][0] for i in range(self.batchSize)]])
                decoderInput = decoderInput.to(self.device)
                maskLoss, nTotal = self.decoder.maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
                loss += maskLoss
                printLosses.append(maskLoss.item() * nTotal)
                nTotals += nTotal

        loss.backward()
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        self.encoderOptimizer.step()
        self.decoderOptimizer.step()

        return sum(printLosses) / nTotals


if __name__ == '__main__':
    counselor = DataPreprocessor()
    counselor.loadData('ConversationalData.csv')
