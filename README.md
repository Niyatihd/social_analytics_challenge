# social_analytics_challenge



## Sentiment Analysis: Comparing 5 News channels


```python
# Dependencies
import json
import pandas as pd
import numpy as np
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns
from config_SA import *
from datetime import datetime
import time
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Search Term
target_channels = ["@CBSNews", "@CNN", "@BBC",
                "@FoxNews", "@nytimes"]
#target_channels = ["@CBSNews"]
```


```python
# Make a function to fetch data from twitter
def news_channels_analysis(target_terms):    
    
    news_dict = {"Source Acc.":[],
                 "Date":[],
                 "Tweet":[],
                 "Compound Score":[],
                 "Pos Score":[],
                 "Neu Score":[],
                 "Neg Score":[]}
    # Loop through list of channels
    for target in target_terms:

        # Variables for holding sentiments
        compound_list = []
        positive_list = []
        negative_list = []
        neutral_list = []

        # Loop through 5 pages fetching 20 tweets each time
        for x in range(5):
            news_tweets = api.user_timeline(target, count=20, page=x)
            
            # Loop through tweets to fetch req'd info about each tweet and append into a news_dict
            for tweet in news_tweets:
                #source_account = target_terms
                raw_date = tweet['created_at']
                converted_date = datetime.strptime(raw_date, "%a %b %d %X +0000 %Y").strftime("%m/%d/%y")
                text = tweet['text']
                source_acc = tweet['user']['screen_name']

                # Run Vader Analysis on each tweet
                analysis = analyzer.polarity_scores(tweet["text"])
                compound = analysis["compound"]
                pos = analysis["pos"]
                neu = analysis["neu"]
                neg = analysis["neg"]

                news_dict["Source Acc."].append(source_acc)
                news_dict["Date"].append(converted_date)
                news_dict["Tweet"].append(text)
                news_dict["Compound Score"].append(compound)
                news_dict["Pos Score"].append(pos)
                news_dict["Neu Score"].append(neu)
                news_dict["Neg Score"].append(neg)
            
        print(" ")
        print("----------------------")
        print(str(target) + ' Done!!')
        print("----------------------")
        print(" ")
        time.sleep(5)
    return news_dict

#news_channels_analysis(target_channels)
```


```python
news_dict_df = pd.DataFrame(news_channels_analysis(target_channels))
news_dict_df
```

     
    ----------------------
    @CBSNews Done!!
    ----------------------
     
     
    ----------------------
    @CNN Done!!
    ----------------------
     
     
    ----------------------
    @BBC Done!!
    ----------------------
     
     
    ----------------------
    @FoxNews Done!!
    ----------------------
     
     
    ----------------------
    @nytimes Done!!
    ----------------------
     





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Date</th>
      <th>Neg Score</th>
      <th>Neu Score</th>
      <th>Pos Score</th>
      <th>Source Acc.</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.8225</td>
      <td>03/25/18</td>
      <td>0.350</td>
      <td>0.650</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>President Trump’s lawyers are threatening Stor...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3036</td>
      <td>03/25/18</td>
      <td>0.074</td>
      <td>0.798</td>
      <td>0.129</td>
      <td>CBSNews</td>
      <td>In his Palm Sunday address, Pope Francis encou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Tonight on @60Minutes, Stormy Daniels tells @a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.2960</td>
      <td>03/25/18</td>
      <td>0.115</td>
      <td>0.885</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>“We’re trying to get people to stop dying." - ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Tonight on @60Minutes, meet Milwaukee @Bucks s...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.4019</td>
      <td>03/25/18</td>
      <td>0.135</td>
      <td>0.802</td>
      <td>0.063</td>
      <td>CBSNews</td>
      <td>A man named "Mad Mike" Hughes launched himself...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.4939</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.873</td>
      <td>0.127</td>
      <td>CBSNews</td>
      <td>"This was all because of the courage and effor...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Adult film star and director Stormy Daniels ta...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>The world's first-ever statue of rock legend D...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.7717</td>
      <td>03/25/18</td>
      <td>0.358</td>
      <td>0.642</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Car bomb kills 5, including driver, near parli...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Marjory Stoneman Douglas student Kyle Kashuv c...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Don Imus: The sun sets on his morning radio sh...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0772</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.925</td>
      <td>0.075</td>
      <td>CBSNews</td>
      <td>Marjory Stoneman Douglas students reflect on M...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.8834</td>
      <td>03/25/18</td>
      <td>0.377</td>
      <td>0.623</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Sen. Mark Warner on past vote against assault ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>WATCH: A giant robotic T-Rex goes up in flames...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.1816</td>
      <td>03/25/18</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Sen. Joni Ernst: "Status quo is not OK" on sch...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Five Parkland student activists joined @FaceTh...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.1280</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.889</td>
      <td>0.111</td>
      <td>CBSNews</td>
      <td>Lawyers Joseph diGenova, Victoria Toensing not...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.2500</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.917</td>
      <td>0.083</td>
      <td>CBSNews</td>
      <td>A colossal Key Lime pie the size of a kiddie p...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Women across the country are running for polit...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.8225</td>
      <td>03/25/18</td>
      <td>0.350</td>
      <td>0.650</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>President Trump’s lawyers are threatening Stor...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.3036</td>
      <td>03/25/18</td>
      <td>0.074</td>
      <td>0.798</td>
      <td>0.129</td>
      <td>CBSNews</td>
      <td>In his Palm Sunday address, Pope Francis encou...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Tonight on @60Minutes, Stormy Daniels tells @a...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.2960</td>
      <td>03/25/18</td>
      <td>0.115</td>
      <td>0.885</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>“We’re trying to get people to stop dying." - ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Tonight on @60Minutes, meet Milwaukee @Bucks s...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.4019</td>
      <td>03/25/18</td>
      <td>0.135</td>
      <td>0.802</td>
      <td>0.063</td>
      <td>CBSNews</td>
      <td>A man named "Mad Mike" Hughes launched himself...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.4939</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.873</td>
      <td>0.127</td>
      <td>CBSNews</td>
      <td>"This was all because of the courage and effor...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Adult film star and director Stormy Daniels ta...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>The world's first-ever statue of rock legend D...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-0.7717</td>
      <td>03/25/18</td>
      <td>0.358</td>
      <td>0.642</td>
      <td>0.000</td>
      <td>CBSNews</td>
      <td>Car bomb kills 5, including driver, near parli...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>-0.7964</td>
      <td>03/25/18</td>
      <td>0.303</td>
      <td>0.614</td>
      <td>0.083</td>
      <td>nytimes</td>
      <td>One Houston suburb devastated by Harvey is sur...</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Using Digital Firm, Brexit Campaigners Skirted...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Slammed in September by Hurricanes Irma and Ma...</td>
    </tr>
    <tr>
      <th>473</th>
      <td>-0.5423</td>
      <td>03/25/18</td>
      <td>0.184</td>
      <td>0.816</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>There was no way around the buzz-kill query: H...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.5106</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.852</td>
      <td>0.148</td>
      <td>nytimes</td>
      <td>Internet companies were built on a model in wh...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.2263</td>
      <td>03/25/18</td>
      <td>0.174</td>
      <td>0.826</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Photos from the #MarchForOurLives protests aro...</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Washington is now consumed by a debate over wh...</td>
    </tr>
    <tr>
      <th>477</th>
      <td>-0.5994</td>
      <td>03/25/18</td>
      <td>0.151</td>
      <td>0.849</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>In one of Sweden's gender-free schools, a girl...</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>City Kitchen: A Crisp Cool-Weather Twist for a...</td>
    </tr>
    <tr>
      <th>479</th>
      <td>-0.4767</td>
      <td>03/25/18</td>
      <td>0.158</td>
      <td>0.769</td>
      <td>0.073</td>
      <td>nytimes</td>
      <td>The U.S. and China are waging a cold war for g...</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.2023</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.859</td>
      <td>0.141</td>
      <td>nytimes</td>
      <td>Greenhouse Gas Emissions Rose Last Year. Here ...</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>“How can this be? How can we have so much info...</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.6124</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.750</td>
      <td>0.250</td>
      <td>nytimes</td>
      <td>There are substantial coverage gaps in traditi...</td>
    </tr>
    <tr>
      <th>483</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Where do these Mets players go when they're hu...</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.4588</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.800</td>
      <td>0.200</td>
      <td>nytimes</td>
      <td>You'll probably need to do a little shopping t...</td>
    </tr>
    <tr>
      <th>485</th>
      <td>0.5565</td>
      <td>03/25/18</td>
      <td>0.112</td>
      <td>0.607</td>
      <td>0.280</td>
      <td>nytimes</td>
      <td>Sharing is caring. It's also become a trend at...</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Can Facebook be fixed? https://t.co/0BKiz8K04B</td>
    </tr>
    <tr>
      <th>487</th>
      <td>-0.4404</td>
      <td>03/25/18</td>
      <td>0.209</td>
      <td>0.791</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Sketch Guy: Resistance Is Futile. To Change Ha...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.2023</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.859</td>
      <td>0.141</td>
      <td>nytimes</td>
      <td>Greenhouse gas emissions rose last year. Here ...</td>
    </tr>
    <tr>
      <th>489</th>
      <td>0.7650</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.720</td>
      <td>0.280</td>
      <td>nytimes</td>
      <td>The Great Australian Bight, a pristine stretch...</td>
    </tr>
    <tr>
      <th>490</th>
      <td>-0.7717</td>
      <td>03/25/18</td>
      <td>0.295</td>
      <td>0.705</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>The Trump administration raised fears of a tra...</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>In a story with parallels to that of the Ameri...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>-0.3612</td>
      <td>03/25/18</td>
      <td>0.276</td>
      <td>0.569</td>
      <td>0.154</td>
      <td>nytimes</td>
      <td>Is it worth making pita at home? Absolutely. h...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>How do you remove 200,000 pounds of trash from...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>There will always be something gray and Dicken...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>Need a dinner idea? Try this salmon https://t....</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.1901</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.923</td>
      <td>0.077</td>
      <td>nytimes</td>
      <td>For those of us pinching our pennies, dining o...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.3182</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>0.859</td>
      <td>0.141</td>
      <td>nytimes</td>
      <td>40% of Americans were obese in 2015 and 2016, ...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.5106</td>
      <td>03/25/18</td>
      <td>0.131</td>
      <td>0.618</td>
      <td>0.251</td>
      <td>nytimes</td>
      <td>France is celebrating the memory and the coura...</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.0000</td>
      <td>03/25/18</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>nytimes</td>
      <td>The Saturday Profile: Shining a Cleansing Ligh...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
#news_dict_df.head()
```


```python
# Save DF to csv file
news_dict_df.to_csv("Newschannel_tweets_df.csv")
```


```python
# Save DF to json file
news_dict_df.to_json("Newschannel_tweets_df.json")
```

### Read the saved json file into DF


```python
news_df = pd.read_json("Newschannel_tweets_df.json").sort_values('Source Acc.')
news_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Date</th>
      <th>Neg Score</th>
      <th>Neu Score</th>
      <th>Pos Score</th>
      <th>Source Acc.</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>288</th>
      <td>-0.4753</td>
      <td>2018-03-22</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>BBC</td>
      <td>Think twice before you throw your kitchen wast...</td>
    </tr>
    <tr>
      <th>201</th>
      <td>-0.7506</td>
      <td>2018-03-25</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.000</td>
      <td>BBC</td>
      <td>Tonight, @regyates meets people whose lives ha...</td>
    </tr>
    <tr>
      <th>202</th>
      <td>0.5719</td>
      <td>2018-03-25</td>
      <td>0.000</td>
      <td>0.837</td>
      <td>0.163</td>
      <td>BBC</td>
      <td>Tonight, @mcgregor_ewan and @McgColin celebrat...</td>
    </tr>
    <tr>
      <th>203</th>
      <td>0.0000</td>
      <td>2018-03-25</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC</td>
      <td>The first ever statue of David Bowie has been ...</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.5267</td>
      <td>2018-03-25</td>
      <td>0.000</td>
      <td>0.815</td>
      <td>0.185</td>
      <td>BBC</td>
      <td>When you're enjoying being single and people j...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a table for tweet number in DF
x = np.arange(1,101)
# Repeat for all news channels
y = np.tile(x,5)
news_df['Tweet No.'] = y

# Rearrange all columns in DF
news_df = news_df[['Tweet No.', 'Source Acc.', 'Tweet', 'Date', 'Compound Score', 'Pos Score', 'Neu Score', 'Neg Score']]
news_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet No.</th>
      <th>Source Acc.</th>
      <th>Tweet</th>
      <th>Date</th>
      <th>Compound Score</th>
      <th>Pos Score</th>
      <th>Neu Score</th>
      <th>Neg Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>288</th>
      <td>1</td>
      <td>BBC</td>
      <td>Think twice before you throw your kitchen wast...</td>
      <td>2018-03-22</td>
      <td>-0.4753</td>
      <td>0.000</td>
      <td>0.872</td>
      <td>0.128</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2</td>
      <td>BBC</td>
      <td>Tonight, @regyates meets people whose lives ha...</td>
      <td>2018-03-25</td>
      <td>-0.7506</td>
      <td>0.000</td>
      <td>0.714</td>
      <td>0.286</td>
    </tr>
    <tr>
      <th>202</th>
      <td>3</td>
      <td>BBC</td>
      <td>Tonight, @mcgregor_ewan and @McgColin celebrat...</td>
      <td>2018-03-25</td>
      <td>0.5719</td>
      <td>0.163</td>
      <td>0.837</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>203</th>
      <td>4</td>
      <td>BBC</td>
      <td>The first ever statue of David Bowie has been ...</td>
      <td>2018-03-25</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>204</th>
      <td>5</td>
      <td>BBC</td>
      <td>When you're enjoying being single and people j...</td>
      <td>2018-03-25</td>
      <td>0.5267</td>
      <td>0.185</td>
      <td>0.815</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
####TEST
a = news_df[news_df['Source Acc.'] == 'BBC']
#a['Compound Score'].mean()
a.head()
print(a['Compound Score'].sum())
```

    6.3870999999999984



```python
####TEST
b = news_df[news_df['Source Acc.'] == 'nytimes']
# b['Compound Score'].mean()
b.head()
print(b['Compound Score'].sum())
```

    -3.7186


### Create the Scatter plot for Compound Score of all the News channels


```python
# Create legend for colors
colors = ['lightblue', 'green', 'red', 'blue', 'yellow']

# Use seaborn to make the scatter plot
ax = sns.lmplot(x='Tweet No.', y='Compound Score', data=news_df, fit_reg=False, 
                hue='Source Acc.', legend=False, size=8, 
                scatter_kws={"s":170,'alpha':1, 'edgecolors':'black', 'linewidths':1},
                palette=colors)

# Make the grid, set x-limit and y-limit
plt.grid()
plt.xlim(100,0)
plt.ylim(-1,1)

# Set scale for all the fonts of the plot
sns.set(font_scale=1.4)

# Make x-axis, y-axis & title labels
plt.title("SENTIMENT ANALYSIS OF MEDIA TWEETS (03/25/2018)")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweets Polarity")

# Set the plot baclground color
sns.set_style("dark")

# Format the legend and plot
plt.legend(loc='upper right', title='City Types')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
```


![png](output_16_0.png)



```python
plt.savefig('News_SentimentAnalysis_ScatterPlot.png')
```

### Create a Bar graph showing Average Compound Score for all News channels


```python
news_df_grouped = news_df.groupby('Source Acc.')
```


```python
avg_comp_score = pd.DataFrame(news_df_grouped['Compound Score'].mean())
avg_comp_score = avg_comp_score.rename(columns = {'Compound Score':'Avg Compound Score'})
avg_comp_score.round(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg Compound Score</th>
    </tr>
    <tr>
      <th>Source Acc.</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BBC</th>
      <td>0.06</td>
    </tr>
    <tr>
      <th>CBSNews</th>
      <td>-0.14</td>
    </tr>
    <tr>
      <th>CNN</th>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>FoxNews</th>
      <td>-0.02</td>
    </tr>
    <tr>
      <th>nytimes</th>
      <td>-0.04</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make the Bar Chart
fig, ax = plt.subplots(figsize=(10,8))

bars = ax.bar(avg_comp_score.index, avg_comp_score['Avg Compound Score'], color=colors, alpha=1, width=1,
       edgecolor='black')

# Format Grid
plt.grid(linestyle = '--', dashes=(5,6))

# Set the axis limits
plt.ylim(-.15,.1)

# Make x-axis, y-axis & title labels
ax.set_title("Overall Media Sentiment Based On Twitter (03/25/2018)", fontsize=18)
#ax.set_xlabel("News Channels", fontsize=15)
ax.set_ylabel("Tweet Polarity", fontsize=16)

# Retrieve an element of a plot and set properties
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize('medium')
    tick.set_fontname('Times New Roman')
    tick.set_color('black')
    tick.set_weight('bold')

# Set bar color per performance
# bars = bars[0].set_color('g')

# Print tumor percent change values on individual Bars
rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    va = 'bottom'

    # If value of bar is negative: Place label below bar
    if y_value < 0:
        y_value = rect.get_height()
        space *= -1
        # Vertically align label at top
        va = 'top'

    # Use Y value as label and format number with one decimal place
    label = "{:.2f}".format(y_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(0, space),          # Vertically shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha='center',                # Horizontally center label
        va=va,
        color='black',
        fontsize=15) 

plt.show()
```


![png](output_21_0.png)



```python
plt.savefig('News_SentimentAnalysis_BarGraph.png')
```
