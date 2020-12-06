import os
import time

import tweepy
from secret import *
from Classification import DepressionClassifier
import ElizaChatbot.eliza


class TwitterBot():

    def __init__(self, uniqueId, username):
        self.id = uniqueId
        self.auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.username = username
        self.classifier = DepressionClassifier(loadMode=True)
        self.chatbot = ElizaChatbot.eliza.Eliza()
        self.chatbot.load('ElizaChatbot' + os.sep + 'doctor.txt')

    '''
    Method to send a Tweet
    
    '''

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

    def startConversation(self, tweetId, username=None, uniqueId=None):
        INIT_MESSAGE = "Hello, I am MedellaAI, an artificial intelligence bot built to help those in need of mental " \
                       "health attention. \n\nI recently detected a potential Tweet that indicated you may want to seek" \
                       " help. If you need support or would like to talk, I am here for you!" \
                       " \n\n https://twitter.com/twitter/statuses/" + str(tweetId)
        if username is None:
            if len(self.api.list_direct_messages()) == 0:
                self.sendDirectMessage(INIT_MESSAGE, username=username, uniqueId=uniqueId)
            elif uniqueId != self.api.list_direct_messages()[0].message_create['target']['recipient_id']:
                self.sendDirectMessage(
                    INIT_MESSAGE, username=username, uniqueId=uniqueId)
        else:
            user = self.api.get_user(username)
            if user.id_str != self.api.list_direct_messages()[0].message_create['target']['recipient_id']:
                self.sendDirectMessage(
                    INIT_MESSAGE, username=username, uniqueId=uniqueId)

    def getMessages(self):
        # Returns Sender ID, message
        listMessages = self.api.list_direct_messages(1)
        print("List Messages Success: {}".format(str(len(listMessages) > 0)))
        if len(listMessages) == 0:
            listMessages = self.api.list_direct_messages()
            print("List Messages Success Trial 2: {}".format(str(len(listMessages) > 0)))

        return int(listMessages[0]._json['message_create']['sender_id']), \
               str(listMessages[0]._json['message_create']['message_data']['text'])

    def main(self):
        print("Following Back Process")
        # First Follow Back All
        self.followBackAll()
        # Analyze top 10 Tweets on feed
        print("Analyzing Twitter Feed")
        userTweet = self.analyzeFeed()
        # Iterate through and check which Tweets are depression based
        for tweetId in userTweet.keys():
            # Start the Initial Conversation
            print("Starting Conversation with {}".format(tweetId))
            self.startConversation(tweetId=tweetId, uniqueId=userTweet[tweetId])
            # Wait For Twitter Servers to Update
            print("Allowing Twitter Servers to Update")
            time.sleep(90)
            # Get most recent message that was sent + the sender
            print("Receiving Messages")
            messageSender, lastMessage = self.getMessages()
            # Start Conversation
            while True:
                # If last message not send by MedellaAI then respond to it
                print("Checking ID Status: {}".format(str(messageSender != self.id)))
                if messageSender != self.id:
                    # Get response
                    response = self.chatbot.respond(lastMessage)
                    print("Sending {} in response to {}".format(response, lastMessage))
                    # If response is one that wants to quit, send final message and quit
                    if response is None:
                        self.sendDirectMessage(self.chatbot.final(), messageSender)
                        break
                    # Else send response
                    self.sendDirectMessage(response, uniqueId=messageSender)
                # Wait for User Response
                time.sleep(90)
                # Get last message + the sender
                messageSender, lastMessage = self.getMessages()


if __name__ == '__main__':
    botInstance = TwitterBot(uniqueId=1317999444177129476, username="MedellaAI")
    botInstance.main()