import twitter_pb2

def decode():

	tweets = twitter_pb2.Tweets()

	deleted_tweet_count = 0
	replies_tweet_count = 0

	with open('twitter.pb', 'rb') as f:
		tweets.ParseFromString(f.read())

		for tweet in tweets.tweets:			
			
			if(tweet.is_delete):
				deleted_tweet_count += 1			
			else:	
				if(tweet.insert.reply_to):							
					replies_tweet_count += 1

	print deleted_tweet_count	
	print replies_tweet_count

if __name__ == "__main__":
	decode()