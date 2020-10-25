import os
import snscrape.modules.twitter as snstwitter
import csv
from AnalyseSentiment.AnalyseSentiment import AnalyseSentiment
import pandas as pd
import langid
from TextProcess import cleanText


def checkExisting(filename, string, tweetList):
    try:
        data = pd.read_csv('Data' + os.sep + filename)
        tweets = data._get_column_array(0)
        return not cleanText(string) in tweets and string not in tweetList
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
                if exclusionTest and langid.classify(tweet.content)[0] == 'en' and checkExisting(filename,
                                                                                                 tweet.content,
                                                                                                 tweetList):
                    # Append to tweet list and increment counter
                    tweetList.append(tweet.content)
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
        tweet = cleanText(tweet)
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


if __name__ == '__main__':
    # Get depression tweets not already in trainData.csv
    # tweets = getTweets("Damn Depression", 15000, sentimentFactor=-0.5, filename='trainData.csv')
    # Append of Test Data
    # saveTweetsToFile(['Tweets', 'Label'], tweets, 'testData.csv', 1, append='w')

    # Get regular tweets not already in trainData.csv
    tweets = getTweets("okay", 15000, sentimentFactor=0.1, filename='trainData.csv')
    # Append of Test Data
    saveTweetsToFile(['Tweets', 'Label'], tweets, 'testData.csv', 0, 'a')
