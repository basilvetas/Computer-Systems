// 1. Find the number of deleted messages in the dataset.
db.tweets.find( { delete: { $exists: true } } ).count()
// 1554

// 2. Find the number of tweets that are replies to another tweet.
db.tweets.find( { in_reply_to_status_id: { $ne : null } } ).count()
// 2531

// 3. Find the five user IDs (field name: uid) that have tweeted the most.
db.tweets.aggregate(
	[
		{
			$match: {
				delete: { $exists: false }
			}
		},
		{
			$group : {
				_id : { uid:"$user.id" },
				tweetCount: { $sum: 1 }
			}
		},		
		{
			$sort: { tweetCount : -1 }
		},
		{
			$limit: 5
		}
	]
)
// { "_id" : { "uid" : 1269521828 }, "tweetCount" : 5 }
// { "_id" : { "uid" : 392695315 }, "tweetCount" : 4 }
// { "_id" : { "uid" : 424808364 }, "tweetCount" : 3 }
// { "_id" : { "uid" : 1706901902 }, "tweetCount" : 3 }
// { "_id" : { "uid" : 578015986 }, "tweetCount" : 2 }