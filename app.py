import streamlit as st
from PIL import Image
from textblob import TextBlob
import nltk
import string
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import datetime as dt
import pandas.util.testing as tm

#Clean the Text
def cleanTweet(tweet):
  #Remove URLs
  #tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
  tweet = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','',tweet)
  #Remove mentions
  tweet = re.sub(r'@[A-Za-z0-9]+','',tweet)
  #Remove the '#' symbol
  tweet = re.sub(r'#','',tweet)
  #Remove the '#' symbol
  tweet = re.sub(r'\$[A-Za-z0-9]+','',tweet)
  #Remove RT
  tweet = re.sub(r'RT[\s]+','',tweet)
  #Remove whitespaces
  tweet = " ".join(tweet.split())
  #Remove stopwords
  stop_words = set(stopwords.words('english'))
  word_tokens = word_tokenize(tweet)
  #replace consecutive non-ASCII characters with a space
  tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
  #filter using NLTK library append it to a string
  filtered_tweet = [w for w in word_tokens if not w in stop_words]
  filtered_tweet = []
  #looping through conditions
  for w in word_tokens:
  #check tokens against stop words and punctuations
      if w not in stop_words and w not in string.punctuation:
        filtered_tweet.append(w)
  return ' '.join(filtered_tweet)


def getSubjectivity(tweet):
  return TextBlob(tweet).sentiment.subjectivity

def getPolarity(tweet):
  return TextBlob(tweet).sentiment.polarity

def sentiment_analyzer_scores(tweet):
    vader = SentimentIntensityAnalyzer()
    return vader.polarity_scores(tweet)

def getAnalysis(score):

  if score < 0 :
    st.error('The tweet is Negative')
  elif score == 0 :
    st.warning('The tweet is Neutral')
  else:
    st.success('The tweet is Positive')
    st.balloons()

#Visualize Sentiment of a sentence in a plot
def visualise_sentiments(data):
  sns.heatmap(pd.DataFrame(data).set_index("Tweet").T,center=0, annot=True, cmap = "PiYG")
  sns.set(rc={'figure.figsize':(20,1)})

#Title
st.write("""
# Stock Market Tweets Sentiment Analysis
** Bachelor's Thesis 2020 - **
*** Antonis Poullis ***
""")

page_bg_img = '''
<style>
body {
background-image: url("https://digitalsynopsis.com/wp-content/uploads/2017/02/beautiful-color-gradients-backgrounds-026-saint-petersburg.png");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.image("https://miro.medium.com/max/1060/1*OXSkEOURJABcEvykfgqwqA.png", use_column_width= True)



input = st.text_area("Enter a Tweet to analyze")
if st.button("Apply Analysis"):
    with st.spinner(f'Preprocessing Tweet...'):
        cleanedTweet = cleanTweet(input)
        st.subheader("Processed Tweet: ")
        st.write(cleanedTweet)
        st.subheader("Analysis Results: ")
        score_textblob =  getPolarity(cleanedTweet)
        score_vader = sentiment_analyzer_scores(cleanedTweet)
        getAnalysis(score_textblob)
        subjectivity = getSubjectivity(cleanedTweet)
        st.write('Polarity Score using Textblob Sentiment Analizer:', score_textblob)
        st.write('Vader Sentiment Analysis Results:', score_vader)
        st.write('Subjectivity:',subjectivity)
        sid = SentimentIntensityAnalyzer()
        visual = visualise_sentiments({
            "Tweet":["Polarity Score"] + cleanedTweet.split(),
            "Sentiment":[sid.polarity_scores(cleanedTweet)["compound"]] + [sid.polarity_scores(word)["compound"] for word in cleanedTweet.split()]
        })
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(visual)
        
