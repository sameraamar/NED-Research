select lead_id, max(delta_rank)  mmm from 
(
	select 
    c1.lead_id, d1.timestamp as lead_time, 
    tr2.rank as lead_rank, c1.tweet_id, 
    c1.timestamp,
    c1.timestamp-d1.timestamp as delta_time,
    tr1.rank as tweet_rank,  
    tr1.rank- tr2.rank as delta_rank
	from clusters c1
	join tweet_id_ranked tr1
	on tr1.tweet_id = c1.tweet_id
	join tweet_id_ranked tr2
	on tr2.tweet_id = c1.lead_Id
	join dataset d1 on d1.tweet_id = c1.lead_id
	#order by c1.lead_id, c1.tweet_id
) as clusters_fast_indication
where delta_time<=3700
group by lead_id
having mmm > 50000
;

#where is_fast != 1
;


select distinct
m.tweet_id
, c.size
, c.users
, c.entropy
, tr.rank
, m.votes
, m.voters
, m.votes/m.voters >= 0.5 AS is_event
, d.timestamp 
, d.created_at
, d.tweet_text
from dataset d 
join (
	select m1.tweet_id, sum(m1.yesno) as votes, count(m1.user) as voters
    from mt_votes m1 
    join bonus_users b
	on m1.user = b.user 
    group by m1.tweet_id
) as m
on m.tweet_id = d.tweet_id
join clusters c on c.lead_Id = m.tweet_id
join tweet_id_ranked tr
on tr.tweet_id = m.tweet_id
#where 
#m.yesno = 1
order by tr.rank asc
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';
