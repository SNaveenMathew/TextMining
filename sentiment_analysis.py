from json import load
from stanfordcorenlp import StanfordCoreNLP

stanfordcorenlp_config = load(open("corenlp.cfg"))
corenlp_location = stanfordcorenlp_config["location"]
stanfordcorenlp = StanfordCoreNLP(corenlp_location, quiet = False, memory = '1g')

def tag_sentiment_nltk(string):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(string)

def tag_sentiment_stanfordcorenlp(string):
    annotated = stanfordcorenlp._request(annotators = "sentiment", data = string)
    sentiment = []
    for sentence in annotated["sentences"]:
        sentiment = sentiment + [sentence["sentiment"]]
    return sentiment
