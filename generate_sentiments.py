from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import os

def analyze_sentiment(mode, read_file, write_file):
    if mode == 'vader':
        analyzer = SentimentIntensityAnalyzer()
    elif mode == "textblob":
        pass
    read_file = open(read_file, "r")
    write_file = open(write_file, "w")
    line = read_file.readline()
    while line:
        if "persona:" in line:
            write_file.write(line)
        else:
            query = line[:line.index("\t")]
            if mode == 'vader':
                scores = analyzer.polarity_scores(query)["compound"]
            elif mode == 'textblob':
                scores = TextBlob(query).sentiment.polarity
            if scores >= 0.75:
                label = "very positive"
            elif scores >= 0.4:
                label = "positive"
            elif scores > -0.4:
                label = "neutral"
            elif scores > -0.75:
                label = "negative"
            else:
                label = "very negative"
            first_space = line.index(" ")
            write_file.write(line[:first_space+1] + label + line[first_space:])
        line = read_file.readline()


# print(os.getcwd())
path = "C:\\Users\\Joie\\Desktop\\6864\\ParlAI\\data\\Persona-Chat\\personachat\\"
mode = "textblob"

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        analyze_sentiment(mode, path+filename, path + "sentiment\\" + filename)