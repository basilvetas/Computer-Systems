from json import loads
from itertools import imap
import twitter_pb2

def decode():

	tweets = twitter_pb2.Tweets()

	deleted_count = 0

	with open('twitter.pb', 'rb') as f:
		tweets.ParseFromString(f.read())
		for tweet in tweets.tweets:
			if(tweet.is_delete):
				deleted_count += 1


	print deleted_count



if __name__ == "__main__":
	decode()