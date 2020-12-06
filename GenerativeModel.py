from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import os
import unicodedata
from io import open
import itertools
import pandas as pd

# Max Length of Bot responses
MAX_LENGTH = 25

# Pad token to make everything same length
PAD_TOKEN = 0
# Start of sentence token
SOS_TOKEN = 1
# End of sentence token
EOS_TOKEN = 2
# min count to trim vocabulary
MIN_COUNT = 3


# Class to build vocabulary
class VOC:
    # Instance variables to keep track of how many of each word + the index
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        self.numWords = 3

    # Add words from a sentence to vocabulary
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # add a word to vocabulary
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.numWords
            self.word2count[word] = 1
            self.index2word[self.numWords] = word
            self.numWords += 1
        else:
            self.word2count[word] += 1

    # Trim words from vocabulary
    def trim(self, minimumCount):
        if self.trimmed:
            return
        self.trimmed = True

        keepWords = []
        # Checks if it follows the minimum count to be on vocab
        for key, value in self.word2count.items():
            if value >= minimumCount:
                keepWords.append(key)

            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
            self.numWords = 3

            for word in keepWords:
                self.addWord(word)


# Data Preprocessor class to process data before it is fed through the transformer network
class DataPreprocessor:
    # Init train file + other instance variables
    def __init__(self, trainFile='ConversationalData.csv'):
        self.trainFile = trainFile
        self.corpusName = self.trainFile.replace('.csv', '')
        self.voc, self.pairs = self.loadPrepareData(dataFile=self.trainFile,
                                                    corpusName=self.corpusName)

        self.trimmedPairs = self.pairs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Visualize Data
    def showData(self, filename, n):
        with open('Data' + os.sep + filename, encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines[:n]:
            print(line)

    # Load data (although I don't I think I use it lol)
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

    # Load unicode to Ascii
    def unicodeToAscii(self, string):
        return ''.join(
            character for character in unicodedata.normalize('NFD', string)
            if unicodedata.category(character) != 'Mn'
        )

    # normalize the string so it can be properly be processed
    def normalizeString(self, string):
        string = str(string)
        string = self.unicodeToAscii(string.lower().strip())
        string = re.sub(r"([.!?])", r"\1", string)
        string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
        string = re.sub(r"\s", r" ", string)
        return string

    # Iterate through vocabulary
    def readVocs(self, filename, corpusName):
        pairs = []
        dataframe = pd.read_csv('Data' + os.sep + filename, delimiter=',')
        for index, row in dataframe.iterrows():
            pairs.append([
                self.normalizeString(row['Response Text']),
                self.normalizeString(row['Initial Text'])
            ])

        voc = VOC(corpusName)
        return voc, pairs

    # Filter pair to see if it is less than the max length
    def filterPair(self, pair):
        return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

    # Filter through ALL pairs
    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    # Add sentences in pairs through vocabulary
    def loadPrepareData(self, corpusName, dataFile):
        voc, pairs = self.readVocs(dataFile, corpusName)
        pairs = self.filterPairs(pairs)
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])

        return voc, pairs

    # Trim rare words in vocabulary
    def trimRareWords(self, pairs):
        self.voc.trim(MIN_COUNT)
        keepPairs = []
        for pair in pairs:
            inputSequence = pair[0]
            outputSequence = pair[1]
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

    # get indexes from sentences (matrices)
    def indexesFromSentences(self, sentence):
        return [self.voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]

    # Zero pad the matrices
    def zeroPadding(self, listE, fillValue=PAD_TOKEN):
        return list(itertools.zip_longest(*listE, fillvalue=fillValue))

    # Get binary matrix for a list
    def binaryMatrix(self, list, value=PAD_TOKEN):
        matrix = []
        for i, seq in enumerate(list):
            matrix.append([])
            for token in seq:
                if token == PAD_TOKEN:
                    matrix[i].append(0)
                else:
                    matrix[i].append(1)

        return matrix

    # Embed input variable to numerical format
    def inputVar(self, listE):
        listE = [self.unicodeToAscii(sentence) for sentence in listE]
        indexesBatch = [self.indexesFromSentences(sentence) for sentence in listE]
        lengths = torch.tensor([len(indexes) for indexes in indexesBatch])
        padList = self.zeroPadding(indexesBatch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Embed output variable to numerical format
    def outputVar(self, listE):
        indexesBatch = [self.indexesFromSentences(sentence) for sentence in listE]
        maxTagetLength = max([len(indexes) for indexes in indexesBatch])
        padList = self.zeroPadding(indexesBatch)
        mask = self.binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, maxTagetLength

    # Turn batch into training data
    def batch2TrainData(self, pairBatch):
        pairBatch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        inputBatch, outputBatch = [], []
        for pair in pairBatch:
            inputBatch.append(pair[0])
            outputBatch.append(pair[0])
        inp, lengths, = self.inputVar(inputBatch)
        output, mask, maxTargetLength = self.outputVar(outputBatch)
        return inp, lengths, output, mask, maxTargetLength


# Encoder Network for transformers
class EncoderRNN(nn.Module):
    def __init__(self, hiddenSize, embedding, nLayers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        # Layers of encoder
        self.nLayers = nLayers

        # Hidden Size of Encoder
        self.hiddenSize = hiddenSize

        # Encoder embedding network
        self.embedding = embedding

        # GRU network for encoder
        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=(0 if nLayers == 1 else dropout), bidirectional=True)

    # Feed forward method for encoder network
    def forward(self, inputSequence, inputLengths, hidden=None):
        embedded = self.embedding(inputSequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        return outputs, hidden


# Attn Class for Network
class Attn(nn.Module):
    def __init__(self, hiddenSize, method='dot'):
        super(Attn, self).__init__()
        # Which method to se for Attn
        self.method = method
        # Hidden Size of Attn
        self.hiddenSize = hiddenSize

        # set method to calculate Attn
        if self.method == 'general':
            self.attn = nn.Linear(self.hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hiddenSize * 2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(hiddenSize))

    # Calculate dot score
    def dotScore(self, hidden, encoderOutput):
        return torch.sum(hidden * encoderOutput, dim=2)

    # Calculate general score
    def generalScore(self, hidden, encoderOutput):
        energy = self.attn(encoderOutput)
        return torch.sum(hidden * energy, dim=2)

    # Concat Score
    def concatScore(self, hidden, encoderOutput):
        energy = self.attn(torch.cat((hidden.expand(encoderOutput.size(0), -1, -1), encoderOutput), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    # Feed forward method for Attn Class
    def forward(self, hidden, encoderOutputs):
        if self.method == 'general':
            attnEnergies = self.generalScore(hidden, encoderOutputs)
        elif self.method == 'concat':
            attnEnergies = self.concatScore(hidden, encoderOutputs)
        else:
            attnEnergies = self.dotScore(hidden, encoderOutputs)

        attnEnergies = attnEnergies.t()

        return F.softmax(attnEnergies, dim=1).unsqueeze(1)


# Decoder Network for Transformer Network
class DecoderRNN(nn.Module):
    def __init__(self, attnModel, embedding, hiddenSize, outputSize, nLayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        # Use GPU is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Attn Model for Decoder
        self.attnModel = attnModel
        # Hidden Size of Decoder
        self.hiddenSize = hiddenSize
        # Output Size of Decoder
        self.outputSize = outputSize
        # Layers of Decoder
        self.nLayers = nLayers
        # Dropout for
        self.dropout = dropout
        # embedding of decoder
        self.embedding = embedding
        # embedding dropout to prevent overfitting
        self.embeddingDropout = nn.Dropout(dropout)
        # GRU dropout
        self.gru = nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=(0 if nLayers == 1 else dropout))
        # Contact layer
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        # output layer
        self.out = nn.Linear(hiddenSize, outputSize)
        # Attn Network
        self.attn = Attn(attnModel, hiddenSize)

    # Feed forward method for decoder network
    def forward(self, inputStep, lastHidden, encoderOutputs):
        embedded = self.embedding(inputStep)
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

    # Calculate NLL Loss for Masks
    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()


# Generative Model to combine everything
class GenerativeModel:
    def __init__(self, encoder, decoder, encoderOptimizer, decoderOptimizer, batchSize, dataProcessor, iterations,
                 searcher,
                 modelName='GenerativeModel',
                 saveDirectory='Models' + os.sep, teacherForcingRatio=1):

        # Use GPU if possible
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder, decoder, and searcher networks
        self.encoder = encoder
        self.decoder = decoder
        self.searcher = searcher

        # Optimizers
        self.encoderOptimizer = encoderOptimizer
        self.decoderOptimizer = decoderOptimizer

        # Data Processor
        self.dataProcessor = dataProcessor

        # Network Settings
        self.batchSize = batchSize
        self.iterations = iterations
        self.teacherForcingRatio = teacherForcingRatio

        # Save Directory + Model Name
        self.saveDirectory = saveDirectory
        self.modelName = modelName

    # Method to train the model
    def train(self, inputVariable, targetVariable, mask, lengths, maxTargetLength, clip, embedding=None,
              maxLength=MAX_LENGTH):

        # Set gradients to 0 before backpropagation
        self.encoderOptimizer.zero_grad()
        self.decoderOptimizer.zero_grad()

        # Send Input + Output vars + mask to GPU
        inputVariable = inputVariable.to(self.device)
        targetVariable = targetVariable.to(self.device)
        mask = mask.to(self.device)

        # Send lengths to CPU
        lengths = lengths.to('cpu')

        # Vars to keep track of losses
        loss = 0
        printLosses = []
        nTotals = 0

        # Get outputs of encoder
        encoderOutputs, encoderHidden = self.encoder(inputVariable, lengths)

        # Format Decoder Inputs
        decoderInput = torch.LongTensor([[SOS_TOKEN for _ in range(self.batchSize)]])
        decoderInput = decoderInput.to(self.device)
        decoderHidden = encoderHidden[:self.decoder.nLayers]

        # Choose whether to use teacher forcing that iteration or not
        useTeacherForcing = random.random() < self.teacherForcingRatio

        # Uses Teacher forcing
        if useTeacherForcing:
            for t in range(maxTargetLength):
                # Get decoder output
                decoderOutput, decoderHidden = self.decoder(
                    decoderInput, decoderHidden, encoderOutputs
                )

                decoderInput = targetVariable[t].view(1, -1)

                # Calculate + Print Loss
                maskLoss, nTotal = self.decoder.maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
                loss += maskLoss
                printLosses.append(maskLoss.item() * nTotal)
                nTotals += nTotal
        # Do not use Teacher Forcing
        else:
            for t in range(maxTargetLength):
                # Get decoder output
                decoderOutput, decoderHidden = self.decoder(
                    decoderInput, decoderHidden, encoderOutputs
                )
                _, topi = decoderOutput.topk(1)
                decoderInput = torch.LongTensor([[topi[i][0] for i in range(self.batchSize)]])
                decoderInput = decoderInput.to(self.device)

                # Calculate + Print Loss
                maskLoss, nTotal = self.decoder.maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
                loss += maskLoss
                printLosses.append(maskLoss.item() * nTotal)
                nTotals += nTotal
        # Backpropagation
        loss.backward()
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Gradient Descent Type Step
        self.encoderOptimizer.step()
        self.decoderOptimizer.step()

        return sum(printLosses) / nTotals

    # Does the train method multiple times
    def trainIterations(self, clip, printEvery=1, saveEvery=500, loadFilename=False):
        # Get training batches
        trainingBatches = [
            self.dataProcessor.batch2TrainData(
                [random.choice(self.dataProcessor.trimmedPairs) for _ in range(self.batchSize)])
            for _ in range(self.iterations)]

        # Initialize
        print("Initializing...")
        startIteration = 1
        printLoss = 0

        # Continue training if file exists
        if loadFilename:
            startIteration = checkpoint['iteration'] + 1

        print("Training")
        # Train network
        for iteration in range(startIteration, self.iterations + 1):
            trainingBatch = trainingBatches[iteration - 1]
            inputVariable, lengths, targetVariable, mask, maxTargetLength = trainingBatch

            loss = self.train(inputVariable=inputVariable, targetVariable=targetVariable, mask=mask, lengths=lengths,
                              maxTargetLength=maxTargetLength, clip=clip)

            # Print Loss
            printLoss += loss
            if iteration % printEvery == 0:
                printLossAverage = printLoss / printEvery
                print("Iteration: {}, Percent Complete: {:.1f}%, Average Loss: {:.4f}".format(iteration,
                                                                                              iteration / self.iterations * 100,
                                                                                              printLossAverage))
                printLoss = 0

            # Save model + model weights
            if iteration % saveEvery == 0:
                directory = os.path.join(self.saveDirectory, self.modelName, self.dataProcessor.corpusName,
                                         '{}-{}_{}'.format(self.encoder.nLayers, self.decoder.nLayers,
                                                           self.encoder.hiddenSize))

                if not os.path.exists(directory):
                    os.makedirs(directory)

                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoderOptimizer.state_dict(),
                    'de_opt': self.decoderOptimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.dataProcessor.voc.__dict__,
                    'embedding': self.encoder.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

    # Had to make this because for some reason my dataprocessor indexesFromSentence was not working :((
    @staticmethod
    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]

    # Evaluate based on Input
    def evaluate(self, sentence, maxLength=MAX_LENGTH):
        try:
            try:
                # Get Indexes Batch based on sentence and VOC
                indexesBatch = [GenerativeModel.indexesFromSentence(self.dataProcessor.voc, sentence)]
                # Get lengths
                lengths = torch.tensor([len(indexes) for indexes in indexesBatch])
                # Get input batches + send to GPU if applicable
                inputBatch = torch.LongTensor(indexesBatch).transpose(0, 1)
                inputBatch = inputBatch.to(self.device)
                # Send Lengths to CPU
                lengths = lengths.to('cpu')
                # Calculate tokens + scores
                tokens, scores = self.searcher(inputBatch, lengths, maxLength)
                # Decode Words and return them
                decodedWords = [self.dataProcessor.voc.index2word[token.item()] for token in tokens]
                return decodedWords
            except KeyError as e:
                # If word not in VOC in then replace it with nothingness
                sentence = sentence.replace(e.args[0], '').strip()
                return self.evaluate(sentence)
        except RecursionError as e:
            # If sentence cannot be processed, return that it cannot understand
            return ['I', 'am', 'not', 'sure', 'what', 'you', 'mean', 'EOS']

    def evaluateInput(self):
        input_sentence = ''
        while (1):
            try:
                # Get input sentence
                inputSentence = input('> ')
                # Check if it is quit case
                if inputSentence == 'q' or inputSentence == 'quit': break
                # Normalize sentence
                inputSentence = self.dataProcessor.normalizeString(inputSentence)
                # Evaluate sentence
                outputWords = self.evaluate(inputSentence)
                # Format and print response sentence
                outputWords[:] = [x for x in outputWords if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(outputWords))
            except KeyError as e:
                print("Error: Encountered unknown word.")


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        # Encoder + Decoder Networks
        self.encoder = encoder
        self.decoder = decoder

    # Feed forward method for GreedySearchDecoder
    def forward(self, inputSequence, inputLength, maxLength):
        encoderOutputs, encoderHidden = self.encoder(inputSequence, inputLength)
        decoderHidden = encoderHidden[:self.decoder.nLayers]
        decoderInput = torch.ones(1, 1, device=self.decoder.device, dtype=torch.long) * SOS_TOKEN
        allTokens = torch.zeros([0], device=self.decoder.device, dtype=torch.long)
        allScores = torch.zeros([0], device=self.decoder.device)

        for _ in range(maxLength):
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutputs)
            decoderScores, decoderInput = torch.max(decoderOutput, dim=1)
            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScores), dim=0)
            decoderInput = torch.unsqueeze(decoderInput, 0)

        return allTokens, allScores


if __name__ == '__main__':
    # Encoder/Decoder Settings
    HIDDEN_SIZE = 400
    MODEL_NAME = 'GenerativeModel'
    ATTN_MODEL = 'dot'
    ENCODER_N_LAYERS = 2
    DECODER_N_LAYERS = 2
    DROPOUT = 0.1
    BATCH_SIZE = 64
    LOAD_FILE = 'Models/RedditData_1500.tar'
    # LOAD_FILE = None

    # Initialize data processor
    dataProcessor = DataPreprocessor(trainFile='ConversationalData.csv')

    if LOAD_FILE:
        checkpoint = torch.load(LOAD_FILE)
        encoderSD = checkpoint['en']
        decoderSD = checkpoint['de']
        encoderOptimizerSD = checkpoint['en_opt']
        decoderOptimizerSD = checkpoint['de_opt']
        embeddingSD = checkpoint['embedding']
        dataProcessor.voc.__dict__ = checkpoint['voc_dict']

    # Initialize Word Embeddings
    embedding = nn.Embedding(dataProcessor.voc.numWords, HIDDEN_SIZE)

    if LOAD_FILE:
        embedding.load_state_dict(embeddingSD)

    # Initialize Encoder and Decoder Networks
    encoder = EncoderRNN(hiddenSize=HIDDEN_SIZE, embedding=embedding, nLayers=ENCODER_N_LAYERS, dropout=DROPOUT)
    decoder = DecoderRNN(attnModel=ATTN_MODEL, embedding=embedding, hiddenSize=HIDDEN_SIZE, nLayers=DECODER_N_LAYERS,
                         dropout=DROPOUT, outputSize=dataProcessor.voc.numWords)

    if LOAD_FILE:
        encoder.load_state_dict(encoderSD)
        decoder.load_state_dict(decoderSD)

    encoder = encoder.to(dataProcessor.device)
    decoder = decoder.to(dataProcessor.device)

    # Generative Model Settings
    CLIP = 50.0
    TEACHER_FORCING_RATIO = 1.0
    LEARNING_RATE = 0.001
    DECODER_LEARNING_RATIO = 5.0
    ITERATIONS = 1500
    PRINT_EVERY = 100
    SAVE_EVERY = 500

    # Dropout in Train Mode
    encoder.train()
    decoder.train()

    # Optimizers
    encoderOptimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoderOptimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARNING_RATIO)

    # Load existing encoder/decoder network
    if LOAD_FILE:
        encoderOptimizer.load_state_dict(encoderOptimizerSD)
        decoderOptimizer.load_state_dict(decoderOptimizerSD)

    searcher = GreedySearchDecoder(encoder, decoder)

    for state in encoderOptimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoderOptimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Initialize Generative Model
    generator = GenerativeModel(encoder=encoder, decoder=decoder, encoderOptimizer=encoderOptimizer,
                                decoderOptimizer=decoderOptimizer, batchSize=BATCH_SIZE, dataProcessor=dataProcessor,
                                iterations=ITERATIONS, searcher=searcher)

    # generator.trainIterations(clip=CLIP, printEvery=PRINT_EVERY, saveEvery=SAVE_EVERY)
    encoder.eval()
    decoder.eval()
    generator.evaluateInput()
