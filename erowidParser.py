# -*- coding: utf-8 -*-
import pandas as ps
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
from nltk import sentiment
import nltk

import os
import sys
import re


def run(dataDir):
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
    print("Starting analysis on text")
    fileList = os.listdir()
    sia = sentiment.vader.SentimentIntensityAnalyzer()
    scoreTotals = {}
    scoringOutput = open('scoredSentences', 'w')
    for i in fileList:
        scoreTotals[i] = 0
        sentTokens = tokenizeText(i)
        for sentence in sentTokens:
            sentenceScore = sia.polarity_scores(sentence)
            scoringOutput.write("{} {}".format(sentence, str(sentence)))
            scoreTotals[i] += sentenceScore.get('compound')

        # filteredTokens = filterStopWords(textTokens, stopSet)
        # print("error proccessing {}".format(i))
    return ps.DataFrame(scoreTotals).transpose()

def tokenizeText(file):
    try:
        textDoc = open(file, 'r')
        textStr = textDoc.read().lower()
        sentTokens = nltk.sent_tokenize(textStr, 'english')
    except:
        print("Error reading {}".format(file))
        sentTokens = ""

    return sentTokens


def generateStopList():
    return set(nltk.corpus.stopwords.words("english"))


def filterStopWords(frequency, stopList):
    stopMatches = []
    for stopWord in stopList:
        # if 'the' in frequency.columns will return true
        # but stopWord representing 'the' will not? WTF
        if stopWord in frequency.columns:
            print("found stop word {}... deleting".format(stopWord))
            stopMatches.append(stopWord)
    return frequency.drop(columns=stopMatches).transpose


def writeDfToFile(frequency):
    frequency.to_csv('wordByFrequency')


def mkExportDir():
    if not os.path.isdir('./builtExperiences'):
        os.mkdir('builtExperiences')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if len(sys.argv) > 2 and sys.argv[2] is 'true':
            parseHtml(sys.argv[1])

        run(sys.argv[1])
    else:
        print("Please pass data directory as argument")

if __name__  == "__main__":
    run()