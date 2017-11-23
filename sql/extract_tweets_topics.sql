select count(distinct user, tweet_id), count(distinct user, tweet_id) / count(distinct user) from mt_votes;

select 
tt.tweet_id, tt.topic_id, 
replace( ttt.title, ',', ' ') as title, 
tt.source, d.created_at, d.timestamp, replace( d.tweet_text, ',', ' ') as tweet_text
From tweet2topic tt join dataset d on d.tweet_id = tt.tweet_id 
left outer join topics  ttt on ttt.topic_id = tt.topic_id
#where tt.tweet_id = 100879898255695872
order by tt.tweet_id asc
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';

select count(*) from tweet2topic where source = 'bonus_users(both yes and no)'
;

select * from tweet2topic where tweet_id = 86675887759691776;

create table thesis_2017.bonus_users AS 
select distinct user  from bonus_users_obsolete
order by user;

select mv.yesno, tr.rank, jj.topic_id, c.*, d.tweet_text from clusters c
join dataset d on d.tweet_id = c.tweet_id
join tweet_id_ranked tr on tr.tweet_id = c.tweet_id
join  relevance_judgments jj on jj.tweet_id = c.tweet_id
left outer join mt_votes mv on mv.tweet_id = c.lead_Id
where c.tweet_id in( 
select tweet_id from relevance_judgments where topic_id = 17
);

select * from relevance_judgments j 
join dataset d on 
d.tweet_id = j.tweet_id 
where j.topic_id = 23;

INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/relevance_judgments2cluster.csv' 
FIELDS TERMINATED BY '\t' 
LINES TERMINATED BY '\n';
#-------------------- get all votes from all users


select * from dataset d 
where (d.tweet_id between 94402345202487297 and 94452127371505665) 
and lower(tweet_text ) like '%oslo%'
and (lower(tweet_text ) like '%bomb%' OR
lower(tweet_text ) like '%terrorist%')
;

select distinct
#min(tr.rank), max(tr.rank)
m.tweet_id
, c.users
, c.entropy
, tr.rank
, m.votes
, m.voters
, d.timestamp 
, d.created_at
, d.tweet_text
from dataset d 
join (
	select m1.tweet_id, sum(m1.yesno) as votes, count(m1.user) as voters
    from mt_votes m1 
    group by m1.tweet_id
) as m
on m.tweet_id = d.tweet_id
join clusters c on c.lead_Id = m.tweet_id
join tweet_id_ranked tr
on tr.tweet_id = m.tweet_id
#where 
#m.yesno = 1
order by tr.rank asc
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2_all_users.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';



# -------------------- get all votes from bonus users
	
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
#m.yesno = 1
order by tr.rank asc
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';


# -------------------- get all votes from bonus users, order by fast growing
	
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
, f.p_20k 
, f.p_40k 
, f.p_60k
, f.p_80k  
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
join clusters_growth_speed_grouped_cumm f on f.lead_id = c.lead_Id
join tweet_id_ranked tr
on tr.tweet_id = m.tweet_id
#m.yesno = 1
#order by tr.rank asc
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.fast_grow.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';


select count(distinct user, tweet_id) from mt_votes;
select user, count(distinct tweet_id) from mt_votes;


select 
lead_id, c.tweet_id as tweet_id, i2t.parent, i2t.parentType, i2t.root_id 
from clusters c
left outer join id2tree i2t
on i2t.tweet_id  = c.tweet_id
;


select parentType, count(*) FRom dataset
group by parentType
;

select * from tweet_id_ranked 
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweet2rank.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';


select * from
(select 
	lead_id, 
    min(timestamp) as start_time, 
    min(timestamp) + 3600 as max_time
from clusters 
group by lead_id) as cc
join clusters c on c.lead_id = cc.lead_id
limit 100;

