from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


class DepressionCounselor:
    def __init__(self):
        self.bot = ChatBot("MedellaAI")
        self.trainer = ChatterBotCorpusTrainer(self.bot)

    def trainBot(self):
        self.trainer.train("chatterbot.corpus.english.conversations")

    def getResponse(self, text):
        return self.bot.get_response(text)


if __name__ == '__main__':
    counselor = DepressionCounselor()
    print(counselor.getResponse('I am good  '))
