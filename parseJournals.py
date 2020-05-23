from bs4 import BeautifulSoup as bs
import re
import os
import logging

logging.basicConfig(filename='webJournalParser.log', filemode='a+', level=logging.DEBUG)
logging.info("Starting pipeline ")

def parseHtml(dataDir):
    logging.info("Parsing html files")
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
            logging.info("problem opening {}".format(i))

def mkExportDir():
    if not os.path.isdir('./builtExperiences'):
        os.mkdir('builtExperiences')


if __name__ == "__main__":
    parseHtml