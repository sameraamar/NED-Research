#SELECT * FROM thesis_2017.clusters_all;


#SET @rank=0;
#create table tweet_id_ranked as
#SELECT @rank:=@rank+1 AS rank, tweet_id from dataset
#order by tweet_id asc;

select g.*, r1.rank as first_rank, r2.rank as last_rank, r1.rank + 10000 as window_Rank 
from 
(select lead_id, min(tweet_id) first_tweet, max(tweet_id) as last_tweet, count(*) as size 
from clusters_all group by lead_id) g
join tweet_id_ranked r1 on r1.tweet_id = g.first_tweet
join tweet_id_ranked r2 on r2.tweet_id = g.last_tweet
#join tweet_id_ranked r3 on r3.rank = r1.rank + 100000
having last_rank > window_rank
;

