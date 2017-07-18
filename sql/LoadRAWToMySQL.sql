
DROP TABLE `thesis_2017`.`dataset_raw`;

CREATE TABLE `thesis_2017`.`dataset_raw` (
tweet_id text,
user text  DEFAULT NULL,
created_at text  DEFAULT NULL,
timestamp text ,
retweets text DEFAULT NULL,
likes text  DEFAULT NULL,
parent text  DEFAULT NULL,
parent_User_Id text  DEFAULT NULL,
parentType text  DEFAULT NULL,
root_id text DEFAULT NULL,
depth text,
time_lag_str text ,
topic_id text,
is_topic text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\analysis_3m\\dataset_full_5m_V1.txt' 
INTO TABLE `thesis_2017`.`dataset_raw`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(`tweet_id`, `user`, `created_at`, `timestamp`, `retweets`, `likes`, `parent`, `parent_User_Id`, `parentType`, `root_id`, `depth`, `time_lag_str`, `topic_id`, `is_topic`)
;



SELECT  * FROM `thesis_2017`.`dataset_raw` limit 10;

COMMIT;

############################################################

DROP TABLE `thesis_2017`.`clusters_raw`;

CREATE TABLE `thesis_2017`.`clusters_raw` (
  `lead_Id` text  ,
  `tweet_id` text   ,
  `timestamp` text DEFAULT NULL,
  `nearest` text,
  `distance` text DEFAULT NULL,
  `entropy` double DEFAULT NULL,
  `users` text DEFAULT NULL,
  `size` text DEFAULT NULL,
  `age` text DEFAULT NULL,
  `score` text,
  `text` text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\full_all_clean.csv' 
INTO TABLE `thesis_2017`.`clusters_raw`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(`lead_Id`, `tweet_id`, @dummy, `timestamp`, `nearest`, `distance`, `entropy`, `users`, `size`, `age`, `score`, @dummy, `text`)
;

#(`lead_Id`, `tweet_id`, `created`, `timestamp`, `nearest`, `distance`, `entropy`, `users`, `size`, `age`, `score`, `topic`, `text`)



SELECT  * FROM `thesis_2017`.`clusters_raw` limit 10;

