import tweepy
from secret import *

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


def postTweet(tweet):
    api.update_status(tweet)


def readDirectMessage(amount):
    return api.list_direct_messages[:amount]


def sendDirectMessage(uniqueId, text):
    api.send_direct_message(uniqueId, text)


def getFollowers(uniqueId):
    return api.followers_ids(uniqueId)


def followAccounts(uniqueId):
    api.create_friendship(uniqueId)

def getFeed():
    api.home_timeline()

# Get all followers of account === DONE
# Follow back accounts === DONE
# Analyze tweets on time line
# DM people with depressive tweets
# Perform Therapy

if __name__ == '__main__':
    print(api.get_user('Samratsahoo2013'))
