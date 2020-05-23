# -*- coding: utf-8 -*-
import pandas as ps
from statistics import mean
from nltk import sentiment
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk.corpus
import nltk

import os
import sys
import logging
from datetime import datetime



def run():
    if len(sys.argv) > 1:
        runAnalysis(sys.argv[1])
    else:
        logging.info("Please pass data directory as argument")


def runAnalysis(datadir):
    os.chdir(datadir + '/builtExperiences')
    textAnalysisFrame = analyizeText()
    writeDfToFile(textAnalysisFrame, 'textAnalysisFrame.tsv')


def analyizeText():
    logging.info("Beginning analyis")
    startTime = datetime.now().microsecond

    fileList = os.listdir()
    totalSentimentScores = {}
    frequencyByExperience = {}
    lengthOfExperience = {}
    for i in fileList:
        print(".")
        sentTokens, wordTokens = tokenizeText(i)
        if sentTokens and wordTokens:
            totalSentimentScores[i] = getAverageSentimentScore(sentTokens)
            normalizedWords = harmonizeWords(wordTokens)
            normalizedAndFilteredWords = removeStopWords(normalizedWords)
            localCount = generateWordFrequencies(normalizedAndFilteredWords)
            frequencyByExperience[i] = localCount
            lengthOfExperience[i] = len(localCount)

    stopTime = datetime.now().microsecond
    logging.info("Main analysis done in {}".format(stopTime - startTime))
    return ps.DataFrame([totalSentimentScores, frequencyByExperience, lengthOfExperience])

def compileTotalFrequencies(wordFrequenciesFrame):
    print("here I am")

def tokenizeText(file):
    try:
        textDoc = open(file, 'r')
        textStr = textDoc.read().lower()
        sentTokens = nltk.sent_tokenize(textStr, 'english')
        wordTokens = nltk.word_tokenize(textStr, 'english')
    except:
        logging.error("Error reading/tokenizing {}".format(file))
        sentTokens = ""
        wordTokens = ""

    return sentTokens, wordTokens


def getAverageSentimentScore(sentTokens):
    sia = sentiment.vader.SentimentIntensityAnalyzer()
    scoreTotals = [0]
    for sentence in sentTokens:
        sentenceScore = sia.polarity_scores(sentence)
        scoreTotals.append(sentenceScore.get('compound'))
    return mean(scoreTotals)


def harmonizeWords(wordTokens):
    taggedWords = nltk.pos_tag(wordTokens)
    pSeries = ps.Series(taggedWords)
    hmzWords = pSeries.apply(lambda x: lemmatizeWords(x))
    return hmzWords


def lemmatizeWords(word):
    lemmatizer = WordNetLemmatizer()
    wordNetPos = get_wordnet_pos(word[1])
    return lemmatizer.lemmatize(word[0], wordNetPos) if wordNetPos else word[0]


def removeStopWords(normalizedWords):
    stops = stopwords.words('english')
    stops.append(".")
    stops.append(",")
    stops.append(":")
    stops.append("..")
    stops.append("'")
    stops.append("'s")
    stopSeries = ps.Series(stops)
    return normalizedWords[~normalizedWords.isin(stopSeries.values)]


def generateWordFrequencies(normalizedWords):
    localCount = normalizedWords.value_counts()
    return localCount.to_dict()


def writeDfToFile(file1, filename):
    file1.to_csv(filename)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


if __name__ == "__main__":
    run()
