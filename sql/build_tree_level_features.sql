SHOW VARIABLES LIKE 'secure_file_priv';

DROP TABLE IF EXISTS thesis_analysis.tree_features00 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features01 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features02 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features03 ;
DROP TABLE IF EXISTS thesis_analysis.id2tree_tmp ;

CREATE TABLE `thesis_analysis`.`id2tree_tmp` (
  `tweet_id` bigint(20) NOT NULL,
  `parent` bigint(20) DEFAULT NULL,
  `parentType` text,
  `root_id` bigint(20) DEFAULT NULL,
  `depth` int(11) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `time_delta` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`tweet_id`),
  KEY `id2tree_idx` (`root_id`,`tweet_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE `thesis_analysis`.`id2tree_tmp` 
ADD INDEX `id2tree_tmp_root_idx` (`root_id` ASC);

INSERT INTO thesis_analysis.id2tree_tmp
SELECT t.* FROM 
thesis_2017.id2tree t 
join thesis_2017.dataset d 
on t.root_id = d.tweet_id;

CREATE TABLE IF NOT EXISTS thesis_analysis.tree_features00 (
  `tweet_id` bigint(20) NOT NULL,
  `parent` bigint(20) DEFAULT NULL,
  `parentType` text,
  `root_id` bigint(20) DEFAULT NULL,
  `depth` int(11) DEFAULT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `time_delta` bigint(20) DEFAULT NULL,
  `likes` int(11) DEFAULT NULL,
  `retweets` int(11) DEFAULT NULL,
  `user` text,
  `topic_id` mediumint(9) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE `thesis_analysis`.`tree_features00` 
ADD PRIMARY KEY (`tweet_id`);
ALTER TABLE `thesis_analysis`.`tree_features00` 
ADD INDEX `tree_features00_root_idx` (`root_id` ASC);



INSERT INTO thesis_analysis.tree_features00 
SELECT c.tweet_id, c.parent, c.parentType, c.root_id, c.depth,  c.timestamp,  c.time_delta, 
		d.likes, d.retweets, d.user, tt.topic_id
 FROM
    thesis_analysis.id2tree_tmp c
    JOIN
    thesis_2017.dataset d ON c.tweet_id = d.tweet_id
    LEFT OUTER JOIN
    thesis_2017.tweet2topic tt ON c.tweet_id = tt.tweet_id
    ;


CREATE TABLE `thesis_analysis`.`tree_features01` (
  `root_id` bigint(20) DEFAULT NULL,
  `tree_size` bigint(21) NOT NULL DEFAULT '0',
  `rtwt_count` decimal(23,0) DEFAULT 0,
  `rtwt_only_count` decimal(23,0) DEFAULT 0,
  `rply_count` decimal(23,0) DEFAULT 0,
  `qwt_only_count` decimal(23,0) DEFAULT 0,
  `original_count` decimal(23,0) DEFAULT 0,
  `likes_avg` decimal(14,4) DEFAULT NULL,
  `likes_std` decimal(14,4) DEFAULT NULL,
  `retweets_avg` decimal(14,4) DEFAULT NULL,
  `retweets_std` decimal(14,4) DEFAULT NULL,
  `event_unknown` int DEFAULT NULL,
  `event_yes` int DEFAULT NULL,
  `event_no` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

ALTER TABLE `thesis_analysis`.`tree_features01` 
CHANGE COLUMN `root_id` `root_id` BIGINT(20) NOT NULL ,
ADD PRIMARY KEY (`root_id`);


INSERT INTO thesis_analysis.tree_features01 SELECT c.root_id,
    COUNT(c.tweet_id) AS tree_size,
    SUM(IF(parentType = '', 1, 0)) AS original_count,
    SUM(IF(parentType = 'rtwt' OR parentType = 'qte', 1, 0)) AS rtwt_count,
    SUM(IF(parentType = 'rtwt', 1, 0)) AS rtwt_only_count,
    SUM(IF(parentType = 'rply', 1, 0)) AS rply_count,
    SUM(IF(parentType = 'qte', 1, 0)) AS qwt_only_count,
    AVG(likes) AS likes_avg,
    stddev(likes) AS likes_std,
    AVG(retweets) AS retweets_avg,
    stddev(retweets) AS retweets_std,
	SUM(IF(topic_id IS NULL, 1, 0)) AS event_unknown, 
	SUM(IF(topic_id > -1, 1, 0)) AS event_yes,
	SUM(IF(topic_id = -1, 1, 0)) AS event_no
FROM
    thesis_analysis.tree_features00 c
GROUP BY c.root_id;

CREATE TABLE thesis_analysis.tree_features03 AS (SELECT 
			'root_id', 'tree_size', 'original_count', 'rtwt_count',
			'rtwt_only_count', 'rply_count', 'qwt_only_count', 'likes_avg', 'likes_std',
            'retweets_avg', 'retweets_std', 'event_unknown', 'event_yes', 'event_no') 
UNION (SELECT 
    root_id, tree_size, original_count, rtwt_count, 
    rtwt_only_count, rply_count, qwt_only_count, likes_avg, likes_std,
    retweets_avg, retweets_std, event_unknown, event_yes, event_no
FROM
    thesis_analysis.tree_features01);


SELECT root_id, tree_size, original_count, rtwt_count, 
    rtwt_only_count, rply_count, qwt_only_count, likes_avg, likes_std,
    retweets_avg, retweets_std, event_unknown, event_yes, event_no
FROM thesis_analysis.tree_features03 
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tree_features.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';


#CREATE TABLE thesis_analysis.tree_features02 AS (SELECT 'tweet_id',
#    'parent', 'parentType', 'root_id', 'depth', 'timestamp', 'time_delta', 'likes', 'retweets', 'user', 'is_event') 
#    UNION (SELECT 
#    tweet_id, parent, parentType, root_id, depth, timestamp, time_delta, likes, retweets, user, is_tweet_event
#FROM
#    thesis_analysis.tree_features00);
#
#

SELECT 
    tweet_id, parent, parentType, root_id, depth, timestamp, time_delta, likes, retweets, user, is_tweet_event 
FROM 
    thesis_analysis.tree_features00 
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tree_tweet_features.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';

