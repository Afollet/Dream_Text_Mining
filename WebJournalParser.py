# -*- coding: utf-8 -*-
import pandas as ps
from statistics import mean
from bs4 import BeautifulSoup as bs
from nltk import sentiment
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import os
import sys
import re


def run():
    if len(sys.argv) > 1:
        if len(sys.argv) > 2 and sys.argv[2] is 'true':
            parseHtml(sys.argv[1])
        runAnalysis(sys.argv[1])
    else:
        print("Please pass data directory as argument")


def runAnalysis(datadir):
    os.chdir(datadir + '/builtExperiences')
    analData = analyizeText()
    writeDfToFile(analData)

def parseHtml(dataDir):
    print("Parsing html files")
    os.chdir(dataDir)
    mkExportDir()
    experienceFiles = os.listdir()
    for i in experienceFiles:
        fileIdMatch = re.search('[0-9]+', i)
        if fileIdMatch:
            fileId = fileIdMatch.group(0)
        else:
            fileId = i
        try:
            fileToParse = open(i, 'r', encoding='utf-8', errors='replace')
            output = open('./builtExperiences/' + fileId, 'w+')
            experience = bs(fileToParse, 'html.parser')
            eBody = experience.body
            report = eBody.find(class_="report-text-surround")
            if report:
                tables = report.find_all('table')
                for i in tables:
                    i.decompose()
                experienceText = report.text
                output.write(experienceText)
                output.flush()
                output.close()
        except:
            print("problem opening {}".format(i))


def analyizeText():
        fileList = os.listdir()
        allScoreTotals = {}
        for i in fileList:
            sentTokens, wordTokens = tokenizeText(i)
            allScoreTotals[i] = getAverageSentimentScore(sentTokens)
            harmonizeWords(wordTokens)
        return ps.DataFrame([allScoreTotals]).transpose()

def tokenizeText(file):
    try:
        textDoc = open(file, 'r')
        textStr = textDoc.read().lower()
        sentTokens = nltk.sent_tokenize(textStr, 'english')
        wordTokens = nltk.word_tokenize(textStr, 'english')
    except:
        print("Error reading/tokenizing {}".format(file))
        sentTokens = ""
        wordTokens = ""

    return sentTokens, wordTokens

def getAverageSentimentScore(sentTokens):
    sia = sentiment.vader.SentimentIntensityAnalyzer()
    scoreTotals = []
    for sentence in sentTokens:
        sentenceScore = sia.polarity_scores(sentence)
        scoreTotals.append(sentenceScore.get('compound'))
    return mean(scoreTotals)

def harmonizeWords(wordTokens):
    taggedWords = nltk.pos_tag(wordTokens)
    lem = WordNetLemmatizer()


def generateStopList():
    return set(nltk.corpus.stopwords.words("english"))


def writeDfToFile(frequency):
    frequency.to_csv('wordByFrequency')


def mkExportDir():
    if not os.path.isdir('./builtExperiences'):
        os.mkdir('builtExperiences')

if __name__  == "__main__":
    run()