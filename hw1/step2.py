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
	print "deleted tweet count:", deleted_tweet_count	
	print "replies tweet count:", replies_tweet_count

	for uid, count in most_tweets:
		print "uid:", uid, "- tweet count:", count

if __name__ == "__main__":
	decode()

# deleted tweet count: 1554
# replies tweet count: 2531
# uid: 1269521828 - tweet count: 5
# uid: 392695315 - tweet count: 4
# uid: 424808364 - tweet count: 3
# uid: 1706901902 - tweet count: 3
# uid: 1471774728 - tweet count: 2