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
  `growth_group` float DEFAULT NULL,
  `growth_group_int` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

insert into clusters_growth_speed
select 
	c.lead_id,
    c.tweet_id,
    c.timestamp,
    d.timestamp as start_time,
    c.size,
    r1.rank as lead_rank,
    r2.rank as tweet_rank,
    (r2.rank-r1.rank)/5000.0 as growth_group,
    floor((r2.rank-r1.rank)/5000.0) as growth_group_int
from clusters c
join tweet_id_ranked r2 on c.tweet_id = r2.tweet_id
join tweet_id_ranked r1 on c.lead_id = r1.tweet_id
join dataset d on d.tweet_id = c.lead_id;

SELECT 
cg.lead_id as lead_id, 
cg.size as size,
cg.growth_group_int as growth_group_int,
count(cg.tweet_id) as group_share_count,
100*count(cg.tweet_id)/5000 group_share_prcnt,
max(timestamp - start_time) as group_time_delta
FROM [new-event-detection:petrovic.clusters_growth_speed] as cg
group by lead_id, size, growth_group_int;

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

