

###################################################

DROP TABLE IF EXISTS `thesis_2017`.`dataset`;

CREATE TABLE `thesis_2017`.`dataset` (
tweet_id bigint(20)  PRIMARY KEY,
user text  DEFAULT NULL,
created_at text  DEFAULT NULL,
timestamp int(11) ,
retweets int(11)  DEFAULT NULL,
likes int(11)  DEFAULT NULL,
parent bigint(20)  DEFAULT NULL,
parent_User_Id text  DEFAULT NULL,
parentType text  DEFAULT NULL,
root_id bigint(20) DEFAULT NULL,
depth int(11),
time_lag_str text ,
tweet_text text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\analysis_3m\\dataset_full_5m_V1.txt' 
INTO TABLE `thesis_2017`.`dataset`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES


(`tweet_id`, `user`, `created_at`, `timestamp`, `retweets`, `likes`, @vparent, `parent_User_Id`, `parentType`, @vroot_id, @vdepth, `time_lag_str`, @dummy, @dummy, `tweet_text`)
SET `parent` = nullif(@vparent,''),
    `root_id` = nullif(@vroot_id,''),
    `depth` = nullif(@vdepth,'');
#(`tweet_id`, `user`, `created_at`, `timestamp`, `retweets`, `likes`, `parent`, `parent_User_Id`, `parentType`, `root_id`, `depth`, `time_lag_str`, topic_id, is_topic)

############################################################
### DIFFUSION TREE


DROP TABLE IF EXISTS `thesis_2017`.`id2tree`;

CREATE TABLE `thesis_2017`.`id2tree` (
tweet_id bigint(20)  PRIMARY KEY,
parent bigint(20)  DEFAULT NULL,
parentType text  DEFAULT NULL,
root_id bigint(20) DEFAULT NULL,
depth int(11),
`timestamp` bigint(20),
time_delta bigint(20)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


CREATE INDEX id2tree_idx
ON `thesis_2017`.`id2tree` (root_id, tweet_id);

LOAD DATA LOCAL INFILE 'c:/data/Thesis/threads_petrovic_all/analysis_3m/id2group_30m_V1.txt' 
INTO TABLE `thesis_2017`.`id2tree`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(tweet_id,@vparent,parenttype,@vdepth,@vroot_id,@vtimestamp,@vtime_delta)
SET `parent` = nullif(@vparent,''),
    `root_id` = nullif(@vroot_id,''),
    `depth` = nullif(@vdepth,''),
    `timestamp` = nullif(@vtimestamp,''),
    `time_delta` = nullif(@vtime_delta,'');
#(`tweet_id`, `user`, `created_at`, `timestamp`, `retweets`, `likes`, `parent`, `parent_User_Id`, `parentType`, `root_id`, `depth`, `time_lag_str`, topic_id, is_topic)

############################################################

DROP TABLE IF EXISTS `thesis_2017`.`clusters`;

CREATE TABLE `thesis_2017`.`clusters` (
  `lead_Id` bigint(20) ,
  `tweet_id` bigint(20)  ,
  `timestamp` int(11) DEFAULT NULL,
  `nearest` text,
  `distance` double DEFAULT NULL,
  `entropy` double DEFAULT NULL,
  `users` int(11) DEFAULT NULL,
  `size` int(11) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  `score` text,
  primary key (`lead_Id`, `tweet_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE INDEX cluster_pk_idx ON thesis_2017.clusters (lead_id, tweet_id);

LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\full_all_clean.csv' 
INTO TABLE `thesis_2017`.`clusters`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(`lead_Id`, `tweet_id`, @dummy, `timestamp`, `nearest`, `distance`, `entropy`, `users`, `size`, `age`, `score`, @dummy, @dummy)
;

#(`lead_Id`, `tweet_id`, `created`, `timestamp`, `nearest`, `distance`, `entropy`, `users`, `size`, `age`, `score`, `topic`, `text`)



COMMIT;


################################################################
############ LABELS & TOPICS
###################################################

DROP TABLE IF EXISTS `thesis_2017`.`tweet2topic`;

CREATE TABLE `thesis_2017`.`tweet2topic` (
tweet_id bigint(20)  PRIMARY KEY,
topic_id MEDIUMINT,
    `source` text
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\events_db\\petrovic\\relevance_judgments_00000000.csv' 
INTO TABLE `thesis_2017`.`tweet2topic`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
(`tweet_id`, `topic_id`, @dummy)
SET `source` = 'petrovic';


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\mt_results\\tweet2topic.txt' 
INTO TABLE `thesis_2017`.`tweet2topic`
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(`tweet_id`, @dummy, @dummy, `topic_id`)
SET `source` = 'mt';





DROP TABLE IF EXISTS `thesis_2017`.`topics`;

CREATE TABLE `thesis_2017`.`topics` (
    `topic_id` MEDIUMINT NOT NULL PRIMARY KEY,
    `source` text,
    `title` TEXT
)  ENGINE=INNODB DEFAULT CHARSET=UTF8;


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\events_db\\petrovic\\README' 
INTO TABLE `thesis_2017`.`topics`
FIELDS TERMINATED BY ': '
LINES TERMINATED BY '\n' 
IGNORE 35 LINES
(`topic_id`,  `title`)
SET `source` = 'petrovic';

LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\mt_results\\topics.txt' 
INTO TABLE `thesis_2017`.`topics`
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
( `title`, `topic_id`)
SET `source` = 'mt';

####################################
DROP VIEW IF EXISTS `thesis_2017`.`tweet2topic_vw`;

CREATE VIEW `thesis_2017`.`tweet2topic_vw` AS
    SELECT 
        `r`.`tweet_id` AS `tweet_id`,
        `r`.`topic_id` AS `topic_id`,
        `r`.`source` AS `mapping_source`,
        `t`.`source` AS `topic_source`,
        `t`.`title` AS `title`,
        `d`.`tweet_text`
    FROM
        (`thesis_2017`.`tweet2topic` `r`
        JOIN `thesis_2017`.`topics` `t` ON ((`t`.`topic_id` = `r`.`topic_id`))
        JOIN `thesis_2017`.`dataset` `d` ON ((`d`.`tweet_id` = `r`.`tweet_id`)));

