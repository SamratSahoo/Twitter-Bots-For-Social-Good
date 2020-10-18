import snscrape.modules.twitter as snstwitter
import csv
from AnalyseSentiment.AnalyseSentiment import AnalyseSentiment
import pandas as pd

"""
 Method to get tweets based on a keyword, the amount, and sentiment factor
"""


def getTweets(keyword, count, sentimentFactor=-0.9):
    # List of tweets to return
    tweetList = []
    # Counts how many tweets based on sentiment factor
    realCounter = 0
    # Analyzes tweet sentiment
    analyzer = AnalyseSentiment()

    # Iterates through twitter tweets based on a certain keyword
    for i, tweet in enumerate(snstwitter.TwitterSearchScraper(keyword).get_items()):

        # Once we have reached our desired amount, quit
        if realCounter > count:
            break

        # For Negative sentiment tweets
        if analyzer.Analyse(tweet.content)['overall_sentiment_score'] < sentimentFactor < 0:
            # Append to tweet list and increment counter
            tweetList.append(tweet.content)
            realCounter += 1
        # For Positive/Neutral sentiment tweets
        elif 0 <= sentimentFactor <= analyzer.Analyse(tweet.content)['overall_sentiment_score']:
            tweetList.append(tweet.content)
            realCounter += 1

    return tweetList


"""
Method to save tweets to a CSV file

LABELS:
0 is Non-Depressive Tweets
1 is Depression Tweets
"""


def saveTweetsToFile(fields, tweets, label=0, append=True):
    # List with tweets and labels
    tweetsWithLabels = []

    # Iterate through tweets to add labels
    for tweet in tweets:
        # Replace return characters and hastags with empty space
        tweet = tweet.replace("\n", ' ').replace('#', '')
        tweetsWithLabels.append([tweet, label])

    if not append:
        # Write to CSV file with utf-8 encoding
        with open('tweetList.csv', 'w', encoding="utf-8") as file:
            # Create a CSV writer
            csvWriter = csv.writer(file)
            # Write the fields (1st row in CSV file)
            csvWriter.writerow(fields)
            # Write the tweets and labels
            csvWriter.writerows(tweetsWithLabels)
            # Close the file
            file.close()
    else:
        with open('tweetList.csv', 'a', encoding="utf-8") as file:
            csvWriter = csv.writer(file)
            # Write the fields (1st row in CSV file)
            csvWriter.writerow(fields)
            # Write the tweets and labels
            csvWriter.writerows(tweetsWithLabels)
            # Close the file
            file.close()

    # Use pandas to remove the empty rows
    df = pd.read_csv('tweetList.csv', sep=',', index_col=0)
    df.to_csv('tweetList.csv')


if __name__ == '__main__':
    tweets = getTweets("Damn Depression", 50)
    saveTweetsToFile(['Tweets, Label'], tweets, 1)
