import csv
import os
import pandas as pd
import praw

from TextProcess import cleanText
from secret import *


class RedditScraper:
    def __init__(self, subreddit, sort='hot', limit=900, mode='w', modComment=False, filename='ConversationalData.csv'):
        self.reddit = praw.Reddit(client_id=REDDIT_PERSONAL_USE_SCRIPT, client_secret=REDDIT_SECRET,
                                  password=REDDIT_PASSWORD, user_agent=REDDIT_APP_NAME,
                                  username=REDDIT_USERNAME)
        self.subreddit = subreddit
        self.sort = sort
        self.limit = limit
        self.mode = mode
        self.modComment = modComment
        self.filename = filename

    def setSort(self):
        if self.sort == 'new':
            return self.reddit.subreddit(self.subreddit).new(limit=self.limit)
        elif self.sort == 'top':
            return self.reddit.subreddit(self.subreddit).top(limit=self.limit)
        elif self.sort == 'hot':
            return self.reddit.subreddit(self.subreddit).hot(limit=self.limit)
        else:
            self.sort = 'hot'
            return self.reddit.subreddit(self.subreddit).hot(limit=self.limit)

    def checkExisting(self, string, file):
        return string in open(file=file).read()

    def saveToCSV(self):
        responseReplies = []
        times = 0
        total = 0
        for submission in self.setSort():
            try:
                title = cleanText(submission.title.replace('\n', ' '))
                comment = cleanText(submission.comments.list()[int(self.modComment)].body.replace('\n', ' '))

                # Filter out shorter responses for better data
                if len(title) > 10 and len(comment) > 35 and \
                        not self.checkExisting(string=title, file='Data' + os.sep + self.filename) \
                        and [title, comment] not in responseReplies:
                    responseReplies.append([title, comment])
                    total += 1
            except Exception as e:
                times += 1
                total += 1
                print(str(e) + " " + str(times) + "/" + str(total))

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
        dataframe = pd.read_csv(path)
        questionText = list(dataframe['questionText'])
        answerText = list(dataframe['answerText'])
        qaPairs = []
        for index, question in enumerate(questionText):
            qaPairs.append([cleanText(question), cleanText(answerText[index])])

        with open("Data" + os.sep + self.filename, self.mode, encoding="utf-8") as file:
            csvWriter = csv.writer(file)
            if self.mode == 'w':
                csvWriter.writerow(["Initial Text", "Response Text"])
            csvWriter.writerows(qaPairs)
            file.close()

        df = pd.read_csv('Data' + os.sep + self.filename, sep=',', index_col=0)
        df.to_csv('Data' + os.sep + self.filename)


if __name__ == '__main__':
    scraper = RedditScraper(subreddit='depression_help', limit=50000, modComment=True, mode='w', sort='hot',
                            filename='CounselChatData.csv')
    scraper.addDataFromSource(
        'https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv')
