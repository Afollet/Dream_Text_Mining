from bs4 import BeautifulSoup as bs
import os
import re
import pandas as ps
# -*- coding: utf-8 -*-

def run():
    #cd to where the data resides
    #os.chdir('/home/afollette/jupyter_notebooks/beautifulSoup/')
    os.mkdir('builtExperiences')
    parseHtml()
    #StopWords  should be in working directory
    stopList = generateStopList()
    #change to build file to separate results from input
    os.chdir('./builtExperiences')
    frequency = collectWordFreq()
    filteredFrequency = filterStopWords(frequency,stopList)
    writeDFToFile(filteredFrequency)

def parseHtml():
    #Set to the direc
    os.mkdir('builtExperiences')
    experienceFiles = os.listdir()
    for i in experienceFiles:
        fileIdMatch = re.search('[0-9]+', i)
        if fileIdMatch:
            fileId = fileIdMatch.group(0)
        else :
            fileId = i
        try:
            fileToParse = open(i, 'r', encoding='utf-8', errors='replace')
            output = open('./builtExperiences/' + fileId,'w+')
            experience = bs(fileToParse, 'html.parser')
            eBody= experience.body
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

def collectWordFreq():
    fileList = os.listdir()
    frequency = {}
    for i in fileList:
        try:
            textDoc = open(i, 'r')
            textStr = textDoc.read().lower()
            word_matches = re.findall(r'\b[a-z]{3,20}\b', textStr)

            for word in word_matches:
                count = frequency.get(word,0)
                frequency[word] = count + 1
        except:
            print("error proccessing {}".format(i))
    return ps.DataFrame([frequency])

def generateStopList():
    stopListFile = open('stopWords', 'r', encoding='utf-8')
    stopList = []
    for line in stopListFile:
        stopList = line.replace('\"', '').strip().split(',')
    return stopList

def filterStopWords(frequency, stopList):
    stopMatches = []
    for stopWord in stopList:
        # if 'the' in frequency.columns will return true
        #but stopWord representing 'the' will not? WTF
        if stopWord in frequency.columns:
            print("found stop word {}... deleting".format(stopWord))
            stopMatches.append(stopWord)
    return frequency.drop(columns=stopMatches).transpose

def writeDFToFile(frequency):
    frequency.to_csv('wordByFrequency')

if __name__  == "__main__":
    run()