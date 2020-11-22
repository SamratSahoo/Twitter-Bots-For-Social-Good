import csv
import os
import pandas as pd
import praw
from secret import *


class RedditScraper:
    def __init__(self, subreddit, sort='hot', limit=900, mode='w'):
        self.reddit = praw.Reddit(client_id=REDDIT_PERSONAL_USE_SCRIPT, client_secret=REDDIT_SECRET,
                                  password=REDDIT_PASSWORD, user_agent=REDDIT_APP_NAME,
                                  username=REDDIT_USERNAME)
        self.subreddit = subreddit
        self.sort = sort
        self.limit = limit
        self.mode = mode

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

    def saveToCSV(self):
        responseReplies = []
        for submission in self.setSort():
            # DO IN THE FUTURE: FILTER SHORTER RESPONSES!!
            responseReplies.append(
                [submission.title.replace('\n', ' '), submission.comments.list()[0].body.replace('\n', ' ')])

        with open("Data" + os.sep + "ConversationalData.csv", self.mode, encoding="utf-8") as file:
            csvWriter = csv.writer(file)
            csvWriter.writerow(["Initial Text", "Response Text"])
            csvWriter.writerows(responseReplies)
            file.close()

        # Use pandas to remove the empty rows
        df = pd.read_csv('Data' + os.sep + "ConversationalData.csv", sep=',', index_col=0)
        df.to_csv('Data' + os.sep + "ConversationalData.csv")


if __name__ == '__main__':
    scraper = RedditScraper(subreddit='depression', limit=10)
    scraper.saveToCSV()
