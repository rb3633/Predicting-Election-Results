# Predicting-Election-Results
Predicting US Presidential Elections at county level based on geo-tagged tweets from Twitter

Abstract

For a long time, forecasting political elections has received a lot of attention. Poll surveys and 
national economic growth are commonly used as predictors in traditional political science election 
forecasting models. However, dense polling is costly. In recent decades, social media has grown 
exponentially, attracting research from a wide range of disciplines. Existing research indicates that 
social media data, particularly Twitter data, may reflect the political landscape and has been used 
to predict global election outcomes. In my machine learning models, I have incorporated Twitter sentiment-based 
support rates as well as GDP data at the county level. The dependent variable is the actual voting 
outcome. Models trained include Logistic Regression, K-NN, Random Forest, and Support Vector Classifier 
(SVC) models on 2012 and 2016 election results and tested them on 2020 data. The resultant models 
achieved an accuracy between 69% and 74%.


# Data Collection
1. Tweets: We used snscrape as a scraper and collected more than 30,000 tweets for each 
political party candidate, using their names as keywords. 
In the case of Hillary and Trump in 2016, we figured out that the number of tweets from 
California counties was very low, so we collected additional tweets from this region for each 
candidate, in order to have more data to visualize on a choropleth map.
2. Demographic data:
a. Collected Twitter users: We used the python library Tweepy to access the Twitter API 
and fetch twitter users’ profile pictures from Twitter. We then ran a pre-trained neural 
network to identify age and gender of these users.
b. Real voters: Real voters’ gender and age were collected for the 2016 & 2020 elections 
via the U.S. Census Bureau website.
3. County-level voting results: Election outcomes at county level were collected from MIT 
Data Lab.
4. Economics Data: We also collected the Year on Year (YoY) change in GDP at the county 
level from the Bureau of Economic Analysis.


# Data Preprocessing & Cleaning
We removed duplicated tweets, which might have been posted by bots. Then, we created functions 
to remove http links and emojis from the tweets so that we could perform sentiment analysis later.
We also used the package langid in Python to detect the language of each tweet, then used 
GoogleTranslator to translate non-English tweets to English.


# Exploratory Data Analysis
In order to further understand and explain the scope of the twitter data collected we decided it 
would be beneficial to understand the demographics of the users behind the tweets in comparison 
to those of the actual US voter demographics.
To this end, we sourced the twitter profile pictures of the users behind the tweets using the library 
package tweepy and then fed these pictures into a pre-trained neural network to identify the age 
and gender of the users. We eliminated pictures where the confidence score of the model was 
below 70% and then plotted the data against the actual voters. The graphs reveal a mismatch 
between the most common age and gender groups in our data and the actual voting population.
In particular, Twitter users were younger, aged 24-34 with more males, whereas the majority of 
the actual voting population is aged 45+ with more females

![Maps](Misc/Picture1.png)

# Sentiment Analysis & Topic Modeling
## A) Sentiment Analysis
To perform Sentiment Analysis on the tweets content, we first combined all the tweets for each 
user by Date. We then used Vader’s SentimentIntensityAnalyzer to compute the compound 
sentiment score for each grouped tweet.
In addition, for each candidate, we calculated the daily average compound sentiment score and 
graphed it against the other (Figure 3 Appendix). We noticed that Hillary's sentiment score went 
up significantly around October 25-26, so we created some word clouds to see what the positive 
tweets mentioned in these days. We removed some common but uninformative words, such as the 
candidates' names, "people", and so on. It turned out to be Hillary's birthday (Figure 4 Appendix).
Similarly, in 2020, Trump's sentiment score dropped significantly between October 5 and 9, which 
coincided with the time Trump had Covid (Figure 5 Appendix). We read some tweets and they 
accused Trump of lying about the virus and being unable to protect the country.
Furthermore, in order to compare the sentiment score of each candidate at the county level, we 
first calculated the Support rate for each of candidate.

We then drew a choropleth map to compare the county-level support rates for each candidate.
In red counties, the Republican candidate (Trump) received a higher support rate, 
whereas in blue counties, the Democratic candidate had a higher support rate (Hillary in 2016 or 
Biden in 2020).

![Maps2](Misc/Picture2.jpeg) 
![Maps3](Misc/Picture3.jpeg)

We then compared our results to those reported by the New York Times and found that our data 
were less consistent with the reality on the West side (e.g., counties in California), where we had 
fewer data.

## B) Topic Modeling
We also wanted to understand the thematic content of the collected tweets to discern any major 
topics or individuals mentioned, for which we performed topic modelling using LDA. We have 
illustrated the topic modelling done for the Romney tweets for the 2012 election and explored how 
we decided on 3 topics being chosen, as with 3 topics, the clusters were maximally distanced from 
each other unlike in the 5 topics version (Figure 8).
Mitt Romney - Words in each topic
One of the key takeaways from the topic modelling is that there is no other major thematic or item 
prevalent in the tweets aside from the candidates Romney and Obama both being mentioned in the 
vast majority of the tweets. Our analysis attributes the sentiment gleaned from such tweets to both 
candidates, but a future improvement on the model would be to associate the sentiment of a tweet 
to the correct candidate alone.

## Machine Learning Models and Results
For the machine learning models, initially we used three dependent variables – Sentiment score of 
Republican party candidate, Sentiment score of Democratic party candidate and YoY % change in 
GDP of the county. The independent variable is the actual election result for a particular county. 
It is assigned value 1 (positive) where the Republican party candidate wins and 0 (negative) where 
Democratic party candidate wins. The overview of the model is displayed in the Appendix. 
Intuitively, it is preferred to take the dependent variables at the national level as we aim to analyze 
the results of presidential elections. But then, we would not have enough data to train our model 
on. Therefore, we decided to build our model at county level so that we have enough data for 
training our model. We scraped geo-tagged tweets from twitter and assigned them to the county 
from where the tweet is tweeted, based on its geo-location and the geo-coordinates of the county. 
Consequently, the independent variable was also taken at the county level. We trained and tuned 
parameters of different classification models on the data of 2012 and 2016 presidential elections 
and tested the models on 2020 elections. The 2012 and 2016 elections data was split in 70:30 to 
form training set and validation set. The model was trained on train set and hyper-parameters were 
tuned on validation set. 
Feature Selection – We used logistic regression to decide whether GDP data is an important 
indicator of the election results. Logistic regression assumes a linear relationship between the input 
and output variables and then applies a sigmoid function to constraint the output between 0 to 1. 
The model predicted the probability that the republican candidate wins at a particular county. We 
included all the three possible variables in the model one-by-one, in-pairs and all together, and 
observed that the p-value is high for ‘Year on Year % change in GDP’

Logistic Regression – We converted the probability output of the model to binary 0,1 values by 
putting a threshold. The threshold was set to maximize the accuracy of the prediction. The model 
has Accuracy: 0.74, Precision: 0.74, Recall: 1, F Score: 0.85 and an AUC of 0.54.

K-Nearest Neighbors Classifier – K-NN uses proximity to make prediction about individual data 
point. The nearest neighbor parameter and threshold were kept at 5 and 0.5 respectively, to 
maximize the accuracy. The model has Accuracy: 0.69, Precision: 0.8, Recall: 0.84, F Score: 0.8 
and an AUC of 0.55.

Random Forest Classifier – Random Forest is ensemble learning algorithm based on decision 
trees. We used Gradient Search CV with random forest classifier to obtain the parameters that 
maximize the in-sample accuracy. The model has Accuracy: 0.74, Precision: 0.75, Recall: 0.97, F 
Score: 0.85 and an AUC of 0.52.

Support Vector Classifier – SVC maps data points to high dimensional space and finds the plane 
that divides the two classes. The model has Accuracy: 0.74, Precision: 0.85, Recall: 1, F Score: 
0.85 and an AUC of 0.51.


## Limitations & Future Improvements
### A) Limitation:
The gender and age distribution of Twitter users differs from the voting population. As a result, 
the calculated sentiment scores may not accurately reflect the voting population's sentiment.
Besides this, tweets were scraped using candidate names as search terms, but a large number of 
tweets mentioned both candidates. Hence, it’s difficult to ascertain the sentiment towards a 
particular candidate.
### B) Future Scope:
We need to collect more tweet data over a longer period of time to potentially capture a more 
accurate representation of the actual voting population. We also need to figure out how to 
effectively assign a tweet to a particular candidate if many candidates are mentioned.
