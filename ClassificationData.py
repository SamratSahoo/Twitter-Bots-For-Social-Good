import os
import snscrape.modules.twitter as snstwitter
import csv
from AnalyseSentiment.AnalyseSentiment import AnalyseSentiment
import pandas as pd
import langid
from TextProcess import cleanText


def checkExisting(string, tweetList):
    try:
        # Get data from both train and test files
        data = pd.read_csv('Data' + os.sep + 'testData.csv')
        data2 = pd.read_csv('Data' + os.sep + 'trainData.csv')

        # Grab tweets
        tweets2 = data2._get_column_array(0)
        tweets = data._get_column_array(0)

        # Check if string is in file or tweetList
        return not cleanText(string) in tweets and string not in tweetList and not cleanText(string) in tweets2
    except IndexError as e:
        print(e)
        return False


"""
 Method to get tweets based on a keyword, the amount, and sentiment factor
"""


def getTweets(keyword, count, sentimentFactor=-0.9, filename='trainData.csv'):
    # List of tweets to return
    tweetList = []
    # Counts how many tweets based on sentiment factor
    realCounter = 0
    # Analyzes tweet sentiment
    analyzer = AnalyseSentiment()

    # Iterates through twitter tweets based on a certain keyword
    for i, tweet in enumerate(snstwitter.TwitterSearchScraper(keyword).get_items()):
        # Once we have reached our desired amount, quit
        if realCounter >= count:
            break

        try:
            exclusionTest = True
            with open('ExclusionList.txt') as f:
                exclusionList = f.readlines()

            newExclusionList = []
            for word in exclusionList:
                word = word.replace('\n', '')
                newExclusionList.append(word)

            # For Negative sentiment tweets OR For Positive/Neutral sentiment tweets
            if analyzer.Analyse(tweet.content)['overall_sentiment_score'] < sentimentFactor < 0 or \
                    0 <= sentimentFactor <= analyzer.Analyse(tweet.content)['overall_sentiment_score']:
                # If tweet has word in exclusion list, set exclusion test to false
                for word in newExclusionList:
                    if tweet.content.lower().find(word) > -1:
                        exclusionTest = False

                # if it passes exclusion constraints
                if exclusionTest \
                        and langid.classify(tweet.content)[0] == 'en' \
                        and checkExisting(filename, tweet.content, tweetList) \
                        and len(cleanText(tweet.content)) > 60:
                    # Append to tweet list and increment counter
                    tweetList.append(cleanText(tweet.content))
                    realCounter += 1

        except AttributeError as exc:
            print(exc)

    return tweetList


"""
Method to save tweets to a CSV file

LABELS:
0 is Non-Depressive Tweets
1 is Depression Tweets
"""


def saveTweetsToFile(fields, tweets, filename, label=0, append='a'):
    # List with tweets and labels
    tweetsWithLabels = []

    # Iterate through tweets to add labels
    for tweet in tweets:
        # Replace return characters and hastags with empty space
        tweetsWithLabels.append([tweet, label])

    # Write to CSV file with utf-8 encoding
    with open('Data' + os.sep + filename, append, encoding="utf-8") as file:
        # Create a CSV writer
        csvWriter = csv.writer(file)
        if append == 'w':
            # Write the fields (1st row in CSV file)
            csvWriter.writerow(fields)
        # Write the tweets and labels
        csvWriter.writerows(tweetsWithLabels)
        # Close the file
        file.close()

    # Use pandas to remove the empty rows
    df = pd.read_csv('Data' + os.sep + filename, sep=',', index_col=0)
    df.to_csv('Data' + os.sep + filename)


def getData(query, file, sentimentFactor, count, appendMode, label):
    # Get tweets based on query
    tweets = getTweets(keyword=query, count=count, sentimentFactor=sentimentFactor, filename=file)
    # Labels for CSV file
    Labels = ['Tweets', 'Label']

    # If it is training file
    if 'train' in file:
        # save tweets to testing file
        saveTweetsToFile(fields=Labels, tweets=tweets, filename=file.replace('train', 'test'), label=label,
                         append=appendMode)
    else:
        # save tweets to training file
        saveTweetsToFile(fields=Labels, tweets=tweets, filename=file.replace('test', 'train'), label=label,
                         append=appendMode)


if __name__ == '__main__':
    # Get depression tweets
    queries = ["Damn Depression", "Stressed and Depressed", "Stressed and want to cry", "depressed and want to cry",
               "I feel miserable today", "I suffer from severe depression", "My depression hurts",
               "I absolutely hate life"]

    # Iterate through queries and get tweets for depression
    for query in queries:
        getData(query=query, file='trainData.csv', sentimentFactor=-0.3,
                count=2000, appendMode='a', label=1)

    # Get regular tweets
    getData(query="the", file='trainData.csv', sentimentFactor=0.2,
            count=15000, appendMode='a', label=0)
