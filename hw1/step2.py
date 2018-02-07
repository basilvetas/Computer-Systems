from json import loads
from itertools import imap
import twitter_pb2

def decode():

	tweets = twitter_pb2.Tweets()

	with open('twitter.pb', 'rb') as f:
		tweets.ParseFromString(f.read())
		for tweet in tweets.tweets:
			print tweet.is_delete






if __name__ == "__main__":
	decode()