SHOW VARIABLES LIKE 'secure_file_priv';

DROP TABLE IF EXISTS clusters_growth_speed ;
CREATE TABLE `clusters_growth_speed` (
  `lead_id` bigint(20) NOT NULL ,
  `tweet_id` bigint(20) NOT NULL ,
  `timestamp` int(11) DEFAULT NULL,
  `size` int(11) DEFAULT NULL,
  `start_time` int(11) DEFAULT NULL,
  `lead_rank` int DEFAULT NULL,
  `tweet_rank` int DEFAULT NULL,
  #`group_growth` int DEFAULT NULL,
  #`group_growth_ratio` float DEFAULT NULL,
  `group_growth_int` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER TABLE `thesis_2017`.`clusters_growth_speed` 
ADD PRIMARY KEY (`lead_id`, `tweet_id`);

insert into clusters_growth_speed
select 
	c.lead_id,
    c.tweet_id,
    c.timestamp,
    c.size,
    d.timestamp as start_time,
    r1.rank as lead_rank,
    r2.rank as tweet_rank,
    #(r2.rank-r1.rank+1) AS group_growth,
    #(r2.rank-r1.rank+1)/5000.0 as group_growth_ratio,
    floor((r2.rank-r1.rank+1)/5000.0) as group_growth_int
from clusters c
join tweet_id_ranked r2 on c.tweet_id = r2.tweet_id
join tweet_id_ranked r1 on c.lead_id = r1.tweet_id
join dataset d on d.tweet_id = c.lead_id;


DROP TABLE IF EXISTS clusters_growth_speed_grouped ;
create table clusters_growth_speed_grouped as
SELECT 
cg.lead_id as lead_id, 
cg.size as size,
cg.start_time as start_time,
cg.group_growth_int,
count(cg.tweet_id) as group_share_count,
100*count(cg.tweet_id)/5000 group_share_prcnt,
max(timestamp - start_time) as group_time_delta
FROM clusters_growth_speed as cg
group by cg.lead_id, cg.size, cg.start_time, cg.group_growth_int;

ALTER TABLE `thesis_2017`.`clusters_growth_speed_grouped` 
ADD PRIMARY KEY (`lead_id`, `group_growth_int`);

DROP TABLE IF EXISTS clusters_growth_speed_grouped_flat ;
CREATE TABLE `clusters_growth_speed_grouped_flat` (
  `lead_id` bigint(20) NOT NULL,
  `size` int(11) DEFAULT NULL,
  `start_time` int(11) DEFAULT NULL,
  `count_05k` int(11) DEFAULT NULL,
  `count_10k` int(11) DEFAULT NULL,
  `count_15k` int(11) DEFAULT NULL,
  `count_20k` int(11) DEFAULT NULL,
  `count_25k` int(11) DEFAULT NULL,
  `count_30k` int(11) DEFAULT NULL,
  `count_35k` int(11) DEFAULT NULL,
  `count_40k` int(11) DEFAULT NULL,
  `count_45k` int(11) DEFAULT NULL,
  `count_50k` int(11) DEFAULT NULL,
  `count_55k` int(11) DEFAULT NULL,
  `count_60k` int(11) DEFAULT NULL,
  `count_65k` int(11) DEFAULT NULL,
  `count_70k` int(11) DEFAULT NULL,
  `count_75k` int(11) DEFAULT NULL,
  `count_80k` int(11) DEFAULT NULL,
  `count_85k` int(11) DEFAULT NULL,
  `count_90k` int(11) DEFAULT NULL,
  `count_95k` int(11) DEFAULT NULL,
  `count_100k` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER TABLE `thesis_2017`.`clusters_growth_speed_grouped_flat` 
ADD PRIMARY KEY (`lead_id`);

insert into clusters_growth_speed_grouped_flat 
select lead_id,	max(size) as size, max(start_time) as start_time,
sum(if(group_growth_int=0, group_share_count, 0)) as count_05k,
sum(if(group_growth_int=1, group_share_count, 0)) as count_10k,
sum(if(group_growth_int=2, group_share_count, 0)) as count_15k,
sum(if(group_growth_int=3, group_share_count, 0)) as count_20k,
sum(if(group_growth_int=4, group_share_count, 0)) as count_25k,
sum(if(group_growth_int=5, group_share_count, 0)) as count_30k,
sum(if(group_growth_int=6, group_share_count, 0)) as count_35k,
sum(if(group_growth_int=7, group_share_count, 0)) as count_40k,
sum(if(group_growth_int=8, group_share_count, 0)) as count_45k,
sum(if(group_growth_int=9, group_share_count, 0)) as count_50k,
sum(if(group_growth_int=10, group_share_count, 0)) as count_55k,
sum(if(group_growth_int=11, group_share_count, 0)) as count_60k,
sum(if(group_growth_int=12, group_share_count, 0)) as count_65k,
sum(if(group_growth_int=13, group_share_count, 0)) as count_70k,
sum(if(group_growth_int=14, group_share_count, 0)) as count_75k,
sum(if(group_growth_int=15, group_share_count, 0)) as count_80k,
sum(if(group_growth_int=16, group_share_count, 0)) as count_85k,
sum(if(group_growth_int=17, group_share_count, 0)) as count_90k,
sum(if(group_growth_int=18, group_share_count, 0)) as count_95k,
sum(if(group_growth_int=19, group_share_count, 0)) as count_100k
from clusters_growth_speed_grouped 
group by lead_id;


DROP TABLE IF EXISTS clusters_growth_speed_grouped_cumm ;
CREATE TABLE `clusters_growth_speed_grouped_cumm` (
  `lead_id` bigint(20) NOT NULL,
  `size` int(11) DEFAULT NULL,
  `start_time` int(11) DEFAULT NULL,
  `count_05k` int(11) DEFAULT NULL,
  `count_10k` int(11) DEFAULT NULL,
  `count_15k` int(11) DEFAULT NULL,
  `count_20k` int(11) DEFAULT NULL,
  `count_25k` int(11) DEFAULT NULL,
  `count_30k` int(11) DEFAULT NULL,
  `count_35k` int(11) DEFAULT NULL,
  `count_40k` int(11) DEFAULT NULL,
  `count_45k` int(11) DEFAULT NULL,
  `count_50k` int(11) DEFAULT NULL,
  `count_55k` int(11) DEFAULT NULL,
  `count_60k` int(11) DEFAULT NULL,
  `count_65k` int(11) DEFAULT NULL,
  `count_70k` int(11) DEFAULT NULL,
  `count_75k` int(11) DEFAULT NULL,
  `count_80k` int(11) DEFAULT NULL,
  `count_85k` int(11) DEFAULT NULL,
  `count_90k` int(11) DEFAULT NULL,
  `count_95k` int(11) DEFAULT NULL,
  `count_100k` int(11) DEFAULT NULL,
  `p_05k` float DEFAULT NULL,
  `p_10k` float DEFAULT NULL,
  `p_15k` float DEFAULT NULL,
  `p_20k` float DEFAULT NULL,
  `p_25k` float DEFAULT NULL,
  `p_30k` float DEFAULT NULL,
  `p_35k` float DEFAULT NULL,
  `p_40k` float DEFAULT NULL,
  `p_45k` float DEFAULT NULL,
  `p_50k` float DEFAULT NULL,
  `p_55k` float DEFAULT NULL,
  `p_60k` float DEFAULT NULL,
  `p_65k` float DEFAULT NULL,
  `p_70k` float DEFAULT NULL,
  `p_75k` float DEFAULT NULL,
  `p_80k` float DEFAULT NULL,
  `p_85k` float DEFAULT NULL,
  `p_90k` float DEFAULT NULL,
  `p_95k` float DEFAULT NULL,
  `p_100k` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE `thesis_2017`.`clusters_growth_speed_grouped_cumm` 
ADD PRIMARY KEY (`lead_id`);

insert into clusters_growth_speed_grouped_cumm
select f.lead_id,
		f.size,
        f.start_time,
		f.count_05k ,
        count_10k ,
        count_15k ,
        count_20k ,
        count_25k ,
        count_30k ,
        count_35k ,
        count_40k ,
        count_45k ,
        count_50k ,
        count_55k ,
        count_60k ,
        count_65k ,
        count_70k ,
        count_75k ,
        count_80k ,
        count_85k ,
        count_90k ,
        count_95k ,
        count_100k,
       100 * count_05k / 5000 as p_05k,
       100 * (count_05k + count_10k) / 10000 as p_10k, 
       100 * (count_05k + count_10k + count_15k) / 15000 as p_15k, 
       100 * (count_05k + count_10k + count_15k + count_20k) / 20000 as p_20k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k) / 25000 as p_25k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k) / 30000 as p_30k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k) / 35000 as p_35k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k) / 40000 as p_40k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k) / 45000 as p_45k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k) / 50000 as p_50k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k) / 55000 as p_55k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k) / 60000 as p_60k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k) / 65000 as p_65k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k) / 70000 as p_70k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k) / 75000 as p_75k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k + count_80k) / 80000 as p_80k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k + count_80k + count_85k) / 85000 as p_85k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k + count_80k + count_85k + count_90k) / 90000 as p_90k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k + count_80k + count_85k + count_90k + count_95k) / 95000 as p_95k, 
       100 * (count_05k + count_10k + count_15k + count_20k + count_25k + count_30k + count_35k + count_40k + count_45k + count_50k + count_55k + count_60k + count_65k + count_70k + count_75k + count_80k + count_85k + count_90k + count_95k + count_100k) / 100000 as p_100k
from clusters_growth_speed_grouped_flat f
#order by lead_id, size
;



DROP TABLE IF EXISTS thesis_analysis.clusters_features00 ;
DROP TABLE IF EXISTS thesis_analysis.clusters_features01 ;
DROP TABLE IF EXISTS thesis_analysis.clusters_features02 ;
DROP TABLE IF EXISTS thesis_analysis.clusters_features03 ;


drop table if exists `clusters_features01`;
drop table IF exists clusters_features01a;


CREATE TABLE `clusters_features00` (
  `lead_Id` bigint(20) NOT NULL,
  `tweet_id` bigint(20) NOT NULL,
  `timestamp` int(11) DEFAULT NULL,
  `nearest` text,
  `distance` double DEFAULT NULL,
  `entropy` double DEFAULT NULL,
  `users` int(11) DEFAULT NULL,
  `size` int(11) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `score` text,
  `topic_id` mediumint(9) DEFAULT NULL,
  KEY `primary_00_idx` (`lead_Id`, `tweet_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE `thesis_analysis`.`clusters_features00` 
ADD PRIMARY KEY (`tweet_id`, `lead_Id`);
ALTER TABLE `thesis_analysis`.`clusters_features00` 
ADD INDEX `features00_tweet_idx` (`tweet_id` ASC);
ALTER TABLE `thesis_analysis`.`clusters_features00` 
ADD INDEX `features00_lead_idx` (`lead_id` ASC);


INSERT INTO thesis_analysis.clusters_features00  
SELECT 	c.*, tt.topic_id
FROM thesis_2017.clusters AS c 
LEFT JOIN thesis_2017.tweet2topic AS tt
ON c.tweet_id = tt.tweet_id  ;


CREATE TABLE `clusters_features01a` (
  `lead_id` bigint(20) NOT NULL ,
  `size` int(11) DEFAULT NULL,
  `entropy` double DEFAULT NULL,
  `users` int(11) DEFAULT NULL,
  `starttime` int(11) DEFAULT NULL,
  `sum_timestamp` decimal(32,0) DEFAULT NULL,
  `event_unknown` int DEFAULT NULL,
  `event_yes` int DEFAULT NULL,
  `event_no` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER TABLE `thesis_analysis`.`clusters_features01a` 
ADD INDEX `primary_01a_idx` (`lead_id` ASC);

INSERT INTO thesis_analysis.clusters_features01a  
SELECT 
        c.lead_id, 
        MAX(size) as size, 
        MAX(entropy) AS entropy, 
        MAX(users) users, 
        MIN(timestamp) AS starttime, 
        SUM(c.timestamp) AS sum_timestamp,

		SUM(IF(topic_id IS NULL, 1, 0)) AS event_unknown, 
		SUM(IF(topic_id > -1, 1, 0)) AS event_yes,
		SUM(IF(topic_id = -1, 1, 0)) AS event_no
    FROM
        thesis_analysis.clusters_features00 AS c 
        GROUP BY c.lead_Id;
        
CREATE TABLE `clusters_features01` (
  `lead_id` bigint(20) NOT NULL ,
  `size` int(11) DEFAULT NULL,
  `entropy` double DEFAULT NULL,
  `users` int(11) DEFAULT NULL,
  `starttime` int(11) DEFAULT NULL,
  `sum_timestamp` decimal(32,0) DEFAULT NULL,
  `time_avg` decimal(37,6) DEFAULT NULL,
  `time_stdv` decimal(37,6) DEFAULT NULL,
  `time_v` decimal(37,6) DEFAULT NULL,
  `time_max` decimal(37,6) DEFAULT NULL,
  `time_geometric_mean` decimal(37,6) DEFAULT NULL,
  #`time_KURTOSIS` decimal(23,0), 
  #`time_SKEWNESS` decimal(23,0), 
  `event_unknown` decimal(23,0) DEFAULT NULL,
  `event_yes` decimal(23,0) DEFAULT NULL,
  `event_no` decimal(23,0) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


ALTER TABLE `thesis_analysis`.`clusters_features01` 
ADD PRIMARY KEY (`lead_id`);

        
INSERT INTO clusters_features01
SELECT f1.lead_id,
		f1.size, 
        f1.entropy, 
        f1.users, 
        f1.starttime, 
        f1.sum_timestamp,
        F1.sum_timestamp/f1.size - f1.starttime as time_avg,
        stddev(c.timestamp - f1.starttime) AS time_stdv,
        variance(c.timestamp - f1.starttime) AS time_v,
        max(c.timestamp - f1.starttime) AS time_max,
        EXP( AVG( LOG( c.timestamp - f1.starttime ) ) ) AS time_geometric_mean,
        #KURTOSIS(  c.timestamp - MIN(timestamp) ) AS time_KURTOSIS,
        #SKEWNESS(  c.timestamp - MIN(timestamp) ) AS time_SKEWNESS
        f1.event_unknown, 
		f1.event_yes,
		f1.event_no
        #c.distance,
 FROM clusters_features00 c JOIN thesis_analysis.clusters_features01a f1 ON c.lead_id = f1.lead_id
 GROUP BY c.lead_id;
        


#INSERT INTO thesis_analysis.clusters_features01  
#SELECT 
#        c.lead_id, MAX(size) as size, MAX(entropy) AS entropy, 
#        MAX(users) users, MIN(timestamp) AS starttime, SUM(c.timestamp) AS sum_timestamp,
#        (SUM(c.timestamp) / MAX(size) - MIN(timestamp)) as time_avg,
#        #stddev(c.timestamp - MIN(timestamp)) AS time_stdv,
#        variance(c.timestamp - MIN(timestamp)) AS time_v,
#        max(timestamp) - MIN(timestamp) AS time_max,
#        EXP( AVG( LOG( c.timestamp - MIN(timestamp) ) ) ) AS time_geometric_mean,
#        KURTOSIS(  c.timestamp - MIN(timestamp) ) AS time_KURTOSIS,
#        SKEWNESS(  c.timestamp - MIN(timestamp) ) AS time_SKEWNESS,
#		SUM(IF(topic_id IS NULL, 1, 0)) AS event_unknown, 
#		SUM(IF(topic_id > -1, 1, 0)) AS event_yes,
#		SUM(IF(topic_id = -1, 1, 0)) AS event_no
 #   FROM
 #       thesis_analysis.clusters_features00 AS c 
 #       GROUP BY c.lead_Id;
        


#CREATE INDEX clusters_features01_idx
#ON thesis_analysis.clusters_features01  (lead_id);


#CREATE TABLE thesis_analysis.clusters_features02 AS
#SELECT e.*, f.size, f.entropy, f.users, (f.sum_timestamp / f.size - e.starttime) as avg_time 
#FROM thesis_analysis.clusters_features01 AS f
#JOIN thesis_2017.clusters e
#        ON f.lead_Id = e.lead_id;

#####################################################

CREATE TABLE thesis_analysis.clusters_features02 AS
SELECT 
	lead_id, 
	entropy, 
    users, 
    size, 
    users/size AS users_to_size,
    time_avg, 
    time_stdv,
    time_v,
    time_max,
    time_geometric_mean,
    event_unknown, 
    event_yes, 
    event_no,
	tt.topic_id AS leader_topic_id
FROM
    thesis_analysis.clusters_features01 c
        LEFT JOIN
    thesis_2017.tweet2topic tt ON c.lead_Id = tt.tweet_id; 



#CREATE TABLE thesis_analysis.clusters_features03 AS
#(SELECT 'lead_id', 'entropy', 'users', 'size', 'users_to_size', 
#	'time_avg', 'time_stdv', 'time_v', 'time_max', 'time_geometric_mean',
#	'event_unknown', 'event_yes', 'event_no', 'leader_topic_id')
#UNION
#(SELECT lead_id, entropy, users, size, users_to_size, 
#		time_avg, time_stdv, time_v, time_max, time_geometric_mean,
#		event_unknown, event_yes, event_no, leader_topic_id
#FROM    thesis_analysis.clusters_features02  );

#ALTER TABLE `thesis_analysis`.`clusters_features03` 
#ADD PRIMARY KEY (`lead_Id`);

SELECT 'lead_id', 'entropy', 'users', 'size', 'users_to_size', 
	'time_avg', 'time_stdv', 'time_v', 'time_max', 'time_geometric_mean',
	'event_unknown', 'event_yes', 'event_no', 'leader_topic_id'
    INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/cluster_features2-header.csv' 
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';

SELECT 
    *
    FROM
    thesis_analysis.clusters_features02
    INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/cluster_features2.csv' 
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';

SELECT 
    *
    FROM
    thesis_analysis.clusters_features02
    WHERE leader_topic_id is not null
    INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/cluster_features2_clean.csv' 
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';


DROP VIEW IF EXISTS thesis_analysis.candidate_labels;

CREATE VIEW thesis_analysis.candidate_labels AS
	(SELECT lead_id, 1000 as topic_id  FROM thesis_analysis.clusters_features02 
	where leader_topic_id is null and event_no=0 and event_yes/size > 0.5 ) 
	union
	(SELECT lead_id, -1 as topic_id FROM thesis_analysis.clusters_features02 
	where leader_topic_id is null and event_yes=0 and event_no/size > 0.5  ) 
	union 
	(SELECT lead_id, 1000 as topic_id  FROM thesis_analysis.clusters_features02 
	where leader_topic_id is null and event_no=0 and event_yes/size <= 0.5 and event_yes/event_unknown > 0.5 )
	union
	(SELECT lead_id, -1 as topic_id FROM thesis_analysis.clusters_features02 
	where leader_topic_id is null and event_yes=0 and event_no/size > 0.5 and event_no/event_unknown > 0.5  ) 
    ;
    
    
#insert into thesis_2017.tweet2topic 	select lead_id as tweet_id, topic_id, 'manual' as `source` from thesis_analysis.candidate_labels;    

