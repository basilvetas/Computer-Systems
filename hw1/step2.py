import twitter_pb2
import operator

def decode():
	tweets = twitter_pb2.Tweets()
	deleted_tweet_count = 0
	replies_tweet_count = 0
	uid_tweet_counts = {}

	with open('twitter.pb', 'rb') as f:
		tweets.ParseFromString(f.read())
		for tweet in tweets.tweets:			
			if(tweet.is_delete):				
				deleted_tweet_count += 1			
			else:	
				if(tweet.insert.reply_to):							
					replies_tweet_count += 1				
				if not tweet.insert.uid in uid_tweet_counts:
					uid_tweet_counts[tweet.insert.uid] = 1
				else:
					uid_tweet_counts[tweet.insert.uid] += 1	

	most_tweets = sorted(uid_tweet_counts.items(), key=operator.itemgetter(1), reverse=True)[:5]	
	print "1. Find the number of deleted messages in the dataset.\n", deleted_tweet_count	
	print "2. Find the number of tweets that are replies to another tweet.\n", replies_tweet_count
	print "3. Find the five user IDs (field name: uid) that have tweeted the most."
	for uid, count in most_tweets:
		print "uid:", uid, "- tweet count:", count

if __name__ == "__main__":
	decode()

# 1. Find the number of deleted messages in the dataset.
# 1554
# 2. Find the number of tweets that are replies to another tweet.
# 2531
# 3. Find the five user IDs (field name: uid) that have tweeted the most.
# uid: 1269521828 - tweet count: 5
# uid: 392695315 - tweet count: 4
# uid: 424808364 - tweet count: 3
# uid: 1706901902 - tweet count: 3
# uid: 1471774728 - tweet count: 2