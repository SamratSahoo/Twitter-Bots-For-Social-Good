import tweepy
from secret import *
from Classification import DepressionClassifier


class TwitterBot():

    def __init__(self, uniqueId, username):
        self.id = uniqueId
        self.auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        self.username = username
        self.classifier = DepressionClassifier(loadMode=True)

    def postTweet(self, tweet):
        self.api.update_status(tweet)

    def readDirectMessage(self, amount):
        return self.api.list_direct_messages[:amount]

    def sendDirectMessage(self, text, username=None, uniqueId=None):
        if uniqueId is None:
            user = self.api.get_user(username)
            self.api.send_direct_message(user.id_str, text)
        else:
            self.api.send_direct_message(uniqueId, text)

    def getFollowers(self):
        return self.api.followers_ids(self.id)

    def followAccount(self, accountId):
        self.api.create_friendship(accountId)

    def getFeed(self, count=10):
        topTweets = list(self.api.home_timeline(count=count))
        tweetDict = {}
        for tweet in topTweets:
            # Key = Tweet text, value = [User ID, Tweet ID]
            tweetDict[tweet.text] = [tweet.user.id, tweet.id]

        return tweetDict

    def followBackAll(self):
        for account in self.getFollowers():
            if not self.api.show_friendship(source_id=self.id, target_id=account)[0].following:
                self.followAccount(account)

    def analyzeFeed(self):
        tweets = self.getFeed()
        usersToDM = {}
        for tweet in tweets.keys():
            if self.classifier.predictDepression(tweet):
                # Key: Tweet ID, Value: User ID
                usersToDM[str(tweets[tweet][1])] = tweets[tweet][0]

        return usersToDM

    def startConversation(self, username=None, uniqueId=None):
        user = self.api.get_user(username)
        print(user.id_str)
        print(self.api.list_direct_messages(user.id_str))
        if user.id_str != self.api.list_direct_messages(user.id_str)[0].message_create['target']['recipient_id']:
            self.sendDirectMessage(
                "Hello, I am MedellaAI, an artificial intelligence bot built to help those in need of mental health"
                " attention. \n\nI recently detected a potential Tweet that indicated you may want to seek help. If you"
                " need support or would like to talk, I am here for you!", username=username, uniqueId=uniqueId)

    def readPreviousMessages(self):
        pass

    def sendNextReply(self, username=None, uniqueId=None):
        pass


if __name__ == '__main__':
    botInstance = TwitterBot(uniqueId=1317999444177129476, username="MedellaAI")
    botInstance.startConversation(username="PCodelaborate")
    # botInstance.postTweet("Hello World! This Tweet was sent from Python!")

# Get all followers of account === DONE
# Follow back accounts === DONE
# Analyze tweets on time line
# DM people with depressive tweets
# Perform Therapy
