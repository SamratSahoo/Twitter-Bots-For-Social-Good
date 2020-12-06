# ISM II Original Work
### Program Preface
This project was made for the Independent Study 
Mentorship Program. The Independent Study Mentorship 
Program is a rigorous research-based program offered 
at Frisco ISD schools for passionate high-achieving 
individuals.

Throughout the course of the program, students 
carry-out a year-long research study where they 
analyze articles and interview local professionals. 
Through the culmination of the research attained, 
students create an original work, presented at 
research showcase, and a final product that will be 
showcased at final presentation night. Through the 
merits of this program, individuals are able to follow 
their passions while increasing their prominence in the 
professional world and growing as a person.

The ISM program is a program in which students carry-on skills that not only last for the year, but for life. ISM gives an early and in-depth introduction to the students' respective career fields and professional world. Coming out of this program, students are better equipped to handle the reality of the professional world while having become significantly more knowledgeable of their respective passions. The ISM program serves as the foundation for the success of individuals in the professional world.

### Project Preface 
Everyday, more than 264 million people globally suffer from depression Additionally, another 40 million suffer from anxiety. Tackling such a massive problem on an individual basis can be one of the most daunting challenges in mental health humanity has faced in the past few decades. It can cost billions and the feasibility of tending to the needs of each and every person is near impossible.
 
The product I am proposing offers a scalable, efficient, and cost-saving approach that leverages one of the most prominent social media platforms today: Twitter. For the original work, I have created a Twitter bot that utilizes the power of linguistics and artificial intelligence to both identify mental health issues such as depression and anxiety while also performing a therapy through talking with the user.

### Navigating Github
* **Commits**: This where you can see a short description of the changes I made, the time I made it and the files I changed.
* **Files**: Below the commits are where you can find my program files with all of my code/other resources
* **ReadME**: The ReadME file is this file! You can find a preface and documentation over the whole project.

### Requirements & Setup
* **Step 1:** Install Python 3.7  (This may work with other versions of Python 3 but Python 3.7 is the only one I have tested)
* **Step 2:** Run ```pip3 install -r requirements.txt```
* **Step 3:** Create a Twitter Developer Account and get Twitter Credentials
* **Step 4:** Create a ```secret.py``` file and add the following variables with their respective variables: ```API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, BEARER_TOKEN```. These can all be found within the Twitter Developer Portal. More info can be found [here](https://developer.twitter.com/en/apply-for-access)
* **Step 5:** In ```TwitterBot.py```, change the following line ```botInstance = TwitterBot(uniqueId=1317999444177129476, username="MedellaAI")``` with your respective Twitter account ID and Username.
* **Step 6:** Create a Reddit Application and add the following Credentials: ```REDDIT_SECRET, REDDIT_PERSONAL_USE_SCRIPT, REDDIT_APP_NAME, REDDIT_USERNAME, REDDIT_PASSWORD```. More info can be found [here](https://github.com/JosephLai241/URS/blob/master/docs/How%20to%20Get%20PRAW%20Credentials.md)
* **Step 7:** Run ```python3 TwitterBot.py```

### Documentation
* ```Data/trainData.csv```: This is the train data for the classification network. The contents of the data are formatted as follows: ```Tweet,Label```
* ```Data/testData.csv```: This is the test data for the classification network. The contents of the data are formatted the same as the train data file; the only difference is this is used to test whether the network actually works or not.
* ```Data/ConversationalData.csv```: This is the training data for the transformer network found in ```GenerativeModel.py```. It contains conversational data that is formatted as follows: ```Query,Response```.
* ```ElizaChatbot/doctor.txt```: This is a text file containing all of the potential responses and decompositions and reassemblies possible which is then used in the ```ElizaChatbot/eliza.py``` file.
* ```ElizaChatbot/eliza.py```: This is the bulk of the retrieval based chatbot that uses a decomposition and reassembly algorithm to create a variety of responses while also being versatile in the amount of topics it can respond to.
* ```Classification.py```: This is the file with the depression classification network which was create with Google's Tensorflow. This network has produced accuracies of 99.6% on testing data.
* ```ClassificationData.py```: This is the Twitter Data Scraper that uses a module known as Sn Scrape to grab thousands of tweets from Twitter. Here we can specify a query sentiment so that it gathers depressive or non-depressive tweets.
* ```ConversationalData.py```: This file hosts the Reddit scraper that gets the data for the deep-learning based chatbot.
* ```ExclusionList.txt```: This text file has a list of words that should be excluded from a tweet if they are found within a tweet; it mostly has tweets regarding politics and weather, both of which are generally associated with depressions but do not refer to mental health depressions.
* ```GenerativeModel.py```: This houses a transformer network for the deep learning-based chatbot. It uses data from ```Data/ConversationalData.csv```.
* ```secret.py``` (NOT SHOWN ON GIT): This houses the credentials for the Twitter API and Reddit API; make sure to keep your credentials safe!
* ```TextProcess.py```: This file has helper methods to clean text like emojis, hashtags, and punctuation so that it can be used with a neural network.
* ```TwitterBot.py```: This file integrates everything together to communicate with Twitter users. It has the Generative Model, the Classification Model, and functions that allow us to utilize the Twitter API to communicate with users on the platform and go through the complete process of diagnosing and providing therapy for depression.

### Portfolio
My research and work for this year can be found at my
[Digital Portfolio.](https://samratsahoo.weebly.com)

### Development Process
Majority of the Development Process was streamed on YouTube. The complete playlist of videos can be found [here.](https://www.youtube.com/playlist?list=PLCa7a7W1cl2jecvwPUhKEPA3C0wUulNuC)

### Thank You
I would just like to give a  special thank you to [Wade Brainerd](https://github.com/wadetb) for 
the creation of his Eliza Chatbot port to Python. This made the development process much easier! 

I would also like to give a special thanks to the following individuals for their contributions
to my research throughout this project.
* Trey Blankenship [Raytheon]
* Won Hwa Kim [UT Arlington]
* Vincent Ng [UT Dallas]
* Abhiramon Rajasekharan [UT Dallas]
