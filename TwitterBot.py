import tweepy
from secret import *


class TwitterBot():

    def __init__(self, uniqueId):
        self.id = uniqueId
        self.auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(self.auth)

    def postTweet(self, tweet):
        self.api.update_status(tweet)

    def readDirectMessage(self, amount):
        return self.api.list_direct_messages[:amount]

    def sendDirectMessage(self, uniqueId, text):
        self.api.send_direct_message(uniqueId, text)

    def getFollowers(self):
        return self.api.followers_ids(self.id)

    def followAccounts(self, accountId):
        self.api.create_friendship(accountId)

    def getFeed(self):
        self.api.home_timeline()


# Get all followers of account === DONE
# Follow back accounts === DONE
# Analyze tweets on time line
# DM people with depressive tweets
# Perform Therapy