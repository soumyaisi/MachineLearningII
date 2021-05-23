# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:23:36 2021

@author: soumya
"""

"""
Creating a Twitter App
To create a Twitter app, you will first need to apply for a developer account. This process along with a detailed explanation can be found at developer.twitter.com.

Once you have acquired a developer account, navigate to developer.twitter.com/en/apps, click the blue button that says, Create a New App, and then complete the form with the following fields:

App Name: What your app will be called
Application Description: How your app will be described to its users
Website URLs: Website associated with app–I recommend using the URL to your Twitter profile
Callback URLs: IMPORTANT enter exactly the following: http://127.0.0.1:1410
Tell us how this app will be used: Be clear and hones
When you’ve completed the required form fields, click the blue Create button at the bottom

Read through and indicate whether you accept the developer terms

"""

#https://cran.r-project.org/web/packages/rtweet/vignettes/auth.html
#https://developer.twitter.com/en/apps
#https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-in-python/
#https://towardsdatascience.com/extracting-data-from-twitter-using-python-5ab67bff553a
#https://www.geeksforgeeks.org/extraction-of-tweets-using-tweepy/

#pip install twitter

import twitter
consumer_key = "<Your Consumer Key Here>"
consumer_secret = "<Your Consumer Secret Here>"
access_token = "<Your Access Token Here>"
access_token_secret = "<Your Access Token Secret Here>"
authorization = twitter.OAuth(access_token, access_token_secret,consumer_key, consumer_secret)

import os
path  = "/home/soumya/spam_twitter"

output_filename = os.path.join(path, "python_tweets.json")


import json
t = twitter.Twitter(auth=authorization)


with open(output_filename, 'a') as output_file:
    search_results = t.search.tweets(q="python", count=100)['statuses']
    for tweet in search_results:
        if 'text' in tweet:
            output_file.write(json.dumps(tweet))
            output_file.write("\n\n")







