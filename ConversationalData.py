import csv
import os
import pandas as pd
import praw

from TextProcess import cleanText
from secret import *


class RedditScraper:
    def __init__(self, subreddit, sort='hot', limit=900, mode='w', modComment=False, filename='ConversationalData.csv'):
        # Initialize Reddit Praw
        self.reddit = praw.Reddit(client_id=REDDIT_PERSONAL_USE_SCRIPT, client_secret=REDDIT_SECRET,
                                  password=REDDIT_PASSWORD, user_agent=REDDIT_APP_NAME,
                                  username=REDDIT_USERNAME)
        # Subreddit name
        self.subreddit = subreddit
        # How you want it to be sorted
        self.sort = sort
        # Number of reddit posts
        self.limit = limit
        # write or append mode
        self.mode = mode
        # If there is a mod comment then don't scrape that
        self.modComment = modComment
        # Get filename to save
        self.filename = filename

    # Set how you want to sort the Reddit Thread
    def setSort(self):
        if self.sort == 'new':
            return self.reddit.subreddit(self.subreddit).new(limit=self.limit)
        elif self.sort == 'top':
            return self.reddit.subreddit(self.subreddit).top(limit=self.limit)
        else:
            return self.reddit.subreddit(self.subreddit).hot(limit=self.limit)

    # Checks if string is in file already
    def checkExisting(self, string, file):
        return string in open(file=file).read()

    # Save pairs to CSV
    def saveToCSV(self):
        responseReplies = []
        times = 0
        total = 0
        # iterate through subreddit
        for submission in self.setSort():
            try:
                # Clean the text
                title = cleanText(submission.title.replace('\n', ' '))
                comment = cleanText(submission.comments.list()[int(self.modComment)].body.replace('\n', ' '))

                # Filter out shorter responses for better data & check it does not already exist
                if len(title) > 10 and len(comment) > 35 and \
                        not self.checkExisting(string=title, file='Data' + os.sep + self.filename) \
                        and [title, comment] not in responseReplies:
                    # Append to response replies list if all holds true
                    responseReplies.append([title, comment])
                    total += 1
            # Catch some random exceptions I am getting--not sure why I am getting them
            except Exception as e:
                times += 1
                total += 1
                print(str(e) + " " + str(times) + "/" + str(total))

        # Write to CSV File
        with open("Data" + os.sep + self.filename, self.mode, encoding="utf-8") as file:
            csvWriter = csv.writer(file)
            if self.mode == 'w':
                csvWriter.writerow(["Initial Text", "Response Text"])
            csvWriter.writerows(responseReplies)
            file.close()

        # Use pandas to remove the empty rows
        df = pd.read_csv('Data' + os.sep + self.filename, sep=',', index_col=0)
        df.to_csv('Data' + os.sep + self.filename)

    def addDataFromSource(self, path):
        # Read CSV source
        dataframe = pd.read_csv(path)

        # Get question and answer text
        questionText = list(dataframe['questionText'])
        answerText = list(dataframe['answerText'])
        qaPairs = []

        # Get QA Pairs to add to CSV
        for index, question in enumerate(questionText):
            qaPairs.append([cleanText(question), cleanText(answerText[index])])

        # Write to file
        with open("Data" + os.sep + self.filename, self.mode, encoding="utf-8") as file:
            csvWriter = csv.writer(file)
            if self.mode == 'w':
                csvWriter.writerow(["Initial Text", "Response Text"])
            csvWriter.writerows(qaPairs)
            file.close()

        # Remove empty lines from CSV
        df = pd.read_csv('Data' + os.sep + self.filename, sep=',', index_col=0)
        df.to_csv('Data' + os.sep + self.filename)


if __name__ == '__main__':
    scraper = RedditScraper(subreddit='depression_help', limit=50000, modComment=True, mode='w', sort='hot',
                            filename='ConversationalData.csv')
    scraper.addDataFromSource(
        'https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv')
