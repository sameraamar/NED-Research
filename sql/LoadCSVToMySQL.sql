

###################################################

DROP TABLE IF EXISTS `thesis_2017`.`tweet_ids`;

CREATE TABLE `thesis_2017`.`tweet_ids` (
tweet_id bigint(20)  PRIMARY KEY,
user text  DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 5.6\\Uploads\\tweet_ids.csv' 
INTO TABLE `thesis_2017`.`tweet_ids`
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n';


DROP TABLE IF EXISTS `mt`.`mt_may30`;

CREATE TABLE `mt`.`mt_may30` (
lead_id bigint(20)  PRIMARY KEY,
entropy double DEFAULT NULL,
users int(11)  DEFAULT NULL,
size int(11)  DEFAULT NULL
#,
#tweet_text text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\DataForMT-EndOfMay\\merged2.file.csv' 
INTO TABLE `mt`.`mt_may30`
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(`lead_id`, `entropy`, `users`, `size`) #, `tweet_text`)
;



DROP TABLE IF EXISTS `thesis_2017`.`dataset`;

CREATE TABLE `thesis_2017`.`dataset` (
tweet_id bigint(20)  PRIMARY KEY,
user text  DEFAULT NULL,
created_at text  DEFAULT NULL,
timestamp int(11) ,
retweets int(11)  DEFAULT NULL,
likes int(11)  DEFAULT NULL,
rtwt_likes int(11)  DEFAULT NULL,
parent bigint(20)  DEFAULT NULL,
parent_User_Id text  DEFAULT NULL,
parentType text  DEFAULT NULL,
#root_id bigint(20) DEFAULT NULL,
#depth int(11),
time_lag_str text ,
tweet_text text
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\analysis_3m\\dataset_full_ALL_RTWT_V1.txt' 
INTO TABLE `thesis_2017`.`dataset`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
IGNORE 1 LINES


(`tweet_id`, `user`, `created_at`, `timestamp`, `retweets`, `likes`, `rtwt_likes`, @vparent, `parent_User_Id`, `parentType`, @dummy, @dummy, `time_lag_str`, `tweet_text`)
SET `parent` = nullif(@vparent,'');
    #`root_id` = nullif(@vroot_id,''),
    #`depth` = nullif(@vdepth,'');
    
    
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
time_delta bigint(20),
tree_size int(10) DEFAULT NULL
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


DROP TABLE IF EXISTS thesis_analysis.id2tree_tmp2;

CREATE TABLE `thesis_analysis`.`id2tree_tmp2` (
root_id bigint(20)  PRIMARY KEY,
tree_size int(10)  DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


INSERT INTO thesis_analysis.id2tree_tmp2 
SELECT c.root_id,
    COUNT(c.tweet_id) AS tree_size
FROM
   `thesis_2017`.`id2tree` c
GROUP BY c.root_id;


UPDATE `thesis_2017`.`id2tree` i
JOIN thesis_analysis.id2tree_tmp2 j ON j.root_id = i.root_id
SET i.tree_size = j.tree_size;

DROP TABLE IF EXISTS thesis_analysis.id2tree_tmp2;

DROP VIEW IF EXISTS `thesis_2017`.`id2tree_vw`;
CREATE 
    ALGORITHM = UNDEFINED 
    DEFINER = `root`@`localhost` 
    SQL SECURITY DEFINER
VIEW `thesis_2017`.`id2tree_vw` AS
    SELECT 
        `t`.`tweet_id` AS `tweet_id`,
        `t`.`tree_size` AS `tree_size`,
        `d`.`user` AS `user`,
        `d`.`created_at` AS `created_at`,
        `d`.`retweets` AS `retweets`,
        `d`.`likes` AS `likes`,
        `t`.`parent` AS `parent`,
        `t`.`parentType` AS `parentType`,
        `t`.`root_id` AS `root_id`,
        `t`.`depth` AS `depth`,
        `t`.`timestamp` AS `timestamp`,
        `t`.`time_delta` AS `time_delta`,
        `d`.`tweet_text` AS `tweet_text`
    FROM
        (`thesis_2017`.`dataset` `d`
        JOIN `thesis_2017`.`id2tree` `t` ON ((`t`.`tweet_id` = `d`.`tweet_id`)));

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

ALTER TABLE `thesis_2017`.`clusters` 
ADD INDEX cluster_pk_idx (lead_id, tweet_id);

ALTER TABLE `thesis_2017`.`clusters` 
ADD INDEX `clusters_lead_id_idx` (`lead_Id` ASC);


ALTER TABLE `thesis_2017`.`clusters` 
ADD INDEX `clusters_tweet_id_idx` (`tweet_Id` ASC);


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

#DROP TABLE IF EXISTS `thesis_2017`.`tweet2topic`;

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





#DROP TABLE IF EXISTS `thesis_2017`.`topics`;

CREATE TABLE `thesis_2017`.`topics` (
    `topic_id` MEDIUMINT NOT NULL PRIMARY KEY,
    `topic_source` text,
    `title` TEXT
)  ENGINE=INNODB DEFAULT CHARSET=UTF8;


LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\events_db\\petrovic\\README' 
INTO TABLE `thesis_2017`.`topics`
FIELDS TERMINATED BY ': '
LINES TERMINATED BY '\n' 
IGNORE 35 LINES
(`topic_id`,  `title`)
SET `topic_source` = 'petrovic';

LOAD DATA LOCAL INFILE 'C:\\data\\Thesis\\threads_petrovic_all\\mt_results\\topics.txt' 
INTO TABLE `thesis_2017`.`topics`
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n' 
IGNORE 1 LINES
( `title`, `topic_id`)
SET `topic_source` = 'mt';

####################################
DROP VIEW IF EXISTS `thesis_2017`.`tweet2topic_vw`;

CREATE VIEW `thesis_2017`.`tweet2topic_vw` AS
     SELECT 
        `r`.`topic_id` AS `topic_id`,
        `r`.`source` AS `mapping_source`,
        `t`.`topic_source` AS `topic_source`,
        `t`.`title` AS `topic_title`,
        `d`.`tweet_id` AS `tweet_id`,
        `d`.`user` AS `user`,
        `d`.`created_at` AS `created_at`,
        `d`.`timestamp` AS `timestamp`,
        `d`.`retweets` AS `retweets`,
        `d`.`likes` AS `likes`,
        `d`.`parent` AS `parent`,
        `d`.`parent_User_Id` AS `parent_User_Id`,
        `d`.`parentType` AS `parentType`,
        `d`.`root_id` AS `root_id`,
        `d`.`depth` AS `depth`,
        `d`.`time_lag_str` AS `time_lag_str`,
        `d`.`tweet_text` AS `tweet_text`
    FROM
        ((`thesis_2017`.`tweet2topic` `r`
        JOIN `thesis_2017`.`topics` `t` ON ((`t`.`topic_id` = `r`.`topic_id`)))
        JOIN `thesis_2017`.`dataset` `d` ON ((`d`.`tweet_id` = `r`.`tweet_id`)));



###########################



DROP TABLE IF EXISTS `thesis_2017`.`mt_votes`;

CREATE TABLE `thesis_2017`.`mt_votes` (
idHitResults int PRIMARY KEY,
idHit int,
user text  DEFAULT NULL,
tweet_id bigint(20),
yesno int
) ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA LOCAL INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 5.6\\Uploads\\amt_30_5_17.csv' 
INTO TABLE `thesis_2017`.`mt_votes`
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(`idHitResults`, `idHit`, `user`, @dummy, `tweet_id`, `yesno`);



#########################


Create table thesis_2017.bonus_users (`user` text, `tweet_id` bigint(20));


insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88612360184541184);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88611487760920576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88611433251733504);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88610485330640896);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88609034114048000);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88606639153885185);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88605301149929473);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88605640892755968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88604718162649088);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88604655227109376);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88604630057091072);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88604109988577281);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88597919179091968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88596346323472384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88594165272821760);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88593745846599680);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88592869245452288);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88584082199552000);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88579351012052994);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88576385613967360);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88570979160297472);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88567422386323456);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88565245576089600);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88563148394733568);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88561411948691456);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88554009031487489);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88544613786320898);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88537873514639360);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88535835082899456);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88535726051966976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88534543245656064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88518709743857664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88485851566321664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88470244586295298);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88469250527866880);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88453358301614081);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88447310102663171);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88439240291127296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88436279116709888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88435188589264896);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88433410179211264);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88424509887086592);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88420655300743168);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88417983550062592);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88417425699258368);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88416364536139776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88415148187983873);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88406197568417792);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88394906481074176);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88391421027028992);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88391119037136896);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88390879953428481);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88388908634746880);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88388883485687808);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88383791588048896);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88380599718518784);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88379899273953280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88378917802610688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88377512706588672);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88374887059697665);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88370495635984384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88368817931161600);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88368092320776192);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88367366685200384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88367127622455296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88364028010827776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88359112311713792);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88358596391342080);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88356067217649664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88355974938755073);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88351600292274178);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88350182604935169);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88348903367385088);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88345921221443584);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88345677939216384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88339189338345472);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88332159710011393);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88330729439772673);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88329563435843584);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88327478841589760);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88327143330811906);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88327067816562688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88325578838646785);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88324689646198784);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88321330021285888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88321170637733888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88320507912523776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88319299965550592);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88318519833403393);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88318276555382784);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88317890666835968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88316535919230976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315814515703809);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315344720113666);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315281822326784);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315231469715457);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315214700875776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88315206303875072);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88314451337560064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88313989976690688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88312979166199808);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88313436332765184);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88312245142032384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88311926370729984);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88311855059177473);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88311712473825281);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88311708279521280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88311334994853888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88310953300598784);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88310923940462593);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88310793925435392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88310718436347905);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88310131225403392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88309883765661696);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88307472019881986);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88307337827336192);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88307199406907392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88302367564513280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88301402870398976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88298697540120576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88298437497470976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88293773439799296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88290761925328897);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88285783261323264);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88284843749806081);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88284822774091776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88284013290205185);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88274672558424065);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88272445412360192);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88269354193530880);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88265839362588672);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88265814196752385);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88264623014412288);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88258893590962176);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88248860836769793);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88248479129931776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88241990562615298);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88225167209275392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88222596092526592);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88207140094877696);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88198415942565888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88170578351104001);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88165197029707776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88139846656339968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88092329411493888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88029062500581377);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 88019969262104576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 87996846064152576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 87990529454907392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 87966592549400577);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91375106277904384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91367330059259904);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91353174262288384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91345616143261697);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91343183413383169);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91342868857360387);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91336191517011969);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91321431769616384);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91305589904375809);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91301894701588480);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91301626282913792);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91287382422327296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91285872451911680);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91262573088997377);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91261474193944576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91260106867613696);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91237151437631488);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91226736976609280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91225847767384065);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91198408651583489);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91197498495995904);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91196307313668096);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91180482175320065);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91179001615368192);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91170067726872576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91168889114865665);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91146961301942272);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91142838292717568);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91134634238279680);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91126170153795585);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91115025871282176);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91101025280339968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91074336919785472);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 90967176659480577);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 90964865602174976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 90918728237195264);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 90916912141312000);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 89118990168883200);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 89107388703051776);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 89001620947484674);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 89001448968437760);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88969987489939456);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88965457641611266);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88954686702497792);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88952350475161601);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88855222960656384);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88760406528626688);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88692840493887488);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88666101810073600);
insert into thesis_2017.bonus_users values ('A2PWE9TKEM99BA', 88626012639862785);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91858575315566596);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91845971419467777);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91813843075993600);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91796252160819200);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91796474454753280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91792707965566976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91794339566596098);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91752497156341760);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91734700753825792);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91656820883595264);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91620393382715392);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91536733795127296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91495365362200576);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91493364695973888);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91492001522008064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91401731728084992);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89737150861549568);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89714161914875904);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89709346816139264);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89705592922443776);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89693676934148096);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89689805583159296);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89682494907092992);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89675716907646976);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89603209982779392);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89529969046323200);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89522473833476096);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89520007561748480);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89517042201403392);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89505612714618880);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89491985429307392);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89441808957968384);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89441012061175808);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89413036019945472);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89412968944635904);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89402822893895680);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89402407666204672);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89401984045682688);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89385173292040193);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89380567908499458);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89335051334062080);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89173012804415489);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 89169074361344000);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90269722645499904);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90257307501477888);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90238823186972672);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90234767307575296);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 93047127810387968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 93007072207192064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92968195195011072);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92968161661550593);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92963870896963584);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92938654711955456);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92934363951529986);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92927720190775297);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92861777318129664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92861307560267776);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92816671798083584);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92757339169497088);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92744177418379265);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92733591003480064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92729866482495488);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92704457397256194);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92704079909896193);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92691488584110080);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92687478842081280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92683586527956992);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92683091608477696);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92682965779353600);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92679262179569664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92676447835131904);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92675508285865985);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92672589058682880);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92672823922929664);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92654897492787200);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92646647275847680);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92643572842627072);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92642931143491585);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92641547014770688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92639365972500480);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92630692143443968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92629593252569088);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92616012100415488);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92614783152558080);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92575847415947265);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92572492002111488);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90224302519095296);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90189229736464384);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90074934952472576);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90056991732539392);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90888021762777088);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90858946855837697);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90827074310373376);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90536320967000064);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90372080406757376);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92530704121999360);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92448793592733696);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92394934505709569);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92358049821507584);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92304945734365184);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92288789266972672);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92273731711418369);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92262067351994368);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92247915778686976);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92247097906176000);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92231952261857280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92230807200088064);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92223223911026688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92217897144946688);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92202780877537280);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92133658726629376);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 92054461912068096);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91997465485651968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91972740063571968);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91942717256507394);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91941320557473792);
insert into thesis_2017.bonus_users values ('A2Q8Q1BE2JQR4C', 91932793512275968);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88593137664131072);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88584082199552000);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88571570582327296);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88569339195826177);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88565245576089600);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88563148394733568);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88556672414527488);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88537873514639360);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88535835082899456);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88535726051966976);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88520580424413184);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88503031452282881);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88443547824562177);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88440393699565568);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88436279116709888);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88433410179211264);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88429761168285696);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88427131310321664);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88417983550062592);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88417425699258368);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88415148187983873);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88406197568417792);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88403857113219072);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88381476307083264);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88379899273953280);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88378917802610688);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88376149549400064);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88368817931161600);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88368092320776192);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88367366685200384);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88364028010827776);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88358596391342080);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88351600292274178);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88350828561313792);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88350182604935169);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88348903367385088);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88345921221443584);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88345677939216384);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88339189338345472);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88338803458183168);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88332159710011393);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88331408933793792);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88324966453477376);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88321170637733888);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88319299965550592);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88318519833403393);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88318276555382784);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88317890666835968);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88316611412504576);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88315344720113666);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88315281822326784);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88315231469715457);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88315063726907394);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88314451337560064);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88310923940462593);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88310131225403392);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88309883765661696);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88307472019881986);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88307199406907392);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88301402870398976);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88290761925328897);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88272688677785600);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88225167209275392);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88113657443127296);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88019969262104576);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 88010704061341696);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 87996846064152576);
insert into thesis_2017.bonus_users values ('A1CE2XPYCDRHVZ', 87966592549400577);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 93681201713385472);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 93680501289795584);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 93505401689079809);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 94160807792885760);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 94186099450187776);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 94145574093340672);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 94129014989524992);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94595031507279872);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94588966552088578);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94455524766121984);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94427448120311809);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94432808415674369);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 94264302252457984);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 90372080406757376);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 90318410088579072);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 90307957895606273);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 90239997587886080);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 90224059228499969);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91902300951547904);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91904414859796480);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91879236440305664);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91859422560792576);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91816166720421888);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91813843075993600);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91752497156341760);
insert into thesis_2017.bonus_users values ('A567OJ58X2WX9', 91726794494984192);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95252790665940993);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95185396580876288);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95222402912493568);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95240891433889792);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95221446619578369);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95235770159337472);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95200844215107584);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95179415499194368);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95177460978696192);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95159228322430976);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95158523670953984);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95133546582253568);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95161925259894785);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95114626081095680);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95173350539792384);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95121748017684480);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95101325951516673);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95096984855257088);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95040915386798081);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94992399859843073);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94989543568191488);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95053083062714369);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94989962986000388);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95033843781869568);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95087757394845696);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 95052869178359808);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94980408353103872);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94941682365247488);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94968899191324674);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94926339597021184);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94922526949511168);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94963052323155968);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94967502488076288);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94919855190454274);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94903199621840897);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94937588699365377);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94931586650345472);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94925840462254080);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94926721274494976);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94895289156120576);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94859436258107392);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94868630155694080);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94834853433970688);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94846790414778369);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94819875582775296);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94868193964867584);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94827945415278593);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94870236565749760);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94865337631260674);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94819053503385600);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94880936260403200);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94832320078557184);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94851999757123585);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94870697960144896);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94876788064403456);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94825701458448384);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94880302899527681);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94823772095389696);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94820387262705664);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94883725443207168);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94869892632817664);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94816746619404288);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94813105950949377);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94812275512328193);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94810962678394880);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94811952517365761);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94810308362768384);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94809658245660672);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94809310126817282);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94808529982066688);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94808538353897473);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94807846302130176);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94808420942757888);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94805363286745088);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94804893503721472);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94799570940334082);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94799034073616385);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94769501987549184);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94771854979497984);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94751848149422081);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94777261433159680);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 94746668192378880);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 87104499503726592);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86864547553615872);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86838559662809088);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86595411644653568);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86386979931029504);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 87948330553970688);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 87946883502325760);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 87905783550984192);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88627942028087296);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88615241654607872);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88563148394733568);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88417425699258368);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88394906481074176);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88378917802610688);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88349381505466368);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88331408933793792);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88314451337560064);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88312979166199808);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 88283002471321601);
insert into thesis_2017.bonus_users values ('A1YSYI926BBOHW', 87975023113027584);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93174236205686784);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93199804670296065);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93167739212021760);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93157194748534786);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93166996836982785);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93142191727316993);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93145157104435200);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93140870504783872);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93139679330828289);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93144045597110272);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93143353545342976);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93067596022284288);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93063124902625280);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93077347774894081);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93064341263355904);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93055168320516096);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93047127810387968);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93026932257587200);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93008393425530881);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92990353740808192);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 93003712594837505);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92974423749046272);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92967440228691968);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92963870896963584);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92963489215295488);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92956157563506688);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92926709338341378);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92902088807419904);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92891158426042368);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92816671798083584);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92744177418379265);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92729866482495488);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92699998843703296);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92683091608477696);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92682965779353600);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92655262384652288);
insert into thesis_2017.bonus_users values ('A3RU7ANMLXJQF6', 92645321896771584);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90368259391623168);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90239997587886080);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 90094073561628673);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 87029408875024385);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86889818247802880);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86752425448452096);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86436049068359680);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 86405615211126784);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91969028133888000);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91906512028569600);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91905639596560384);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91837796741947392);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91776576655605761);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91715931243425793);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91652785975730178);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93047127810387968);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93043923362136064);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93041461318270978);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91577351435063296);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93026932257587200);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93026584117788672);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93018208101081088);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93017939648851969);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93016924648243201);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93013502091984896);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93013120401948672);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93012797461512192);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93010943562350592);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93010779988701184);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93010373149605888);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93007881712054272);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93006468235800576);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93006703116824577);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93007072207192064);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93001984508039168);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92993113584443392);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 93003712594837505);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92988495647346688);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91535823614390272);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92998796870549504);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92977443660509184);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92975744971587584);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92972506939527168);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92971600978259968);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92973433889112064);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92974188872220672);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92968161661550593);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92967440228691968);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92967360528519169);
insert into thesis_2017.bonus_users values ('A3RM8SGGQGR3TH', 91512083878916096);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92964130952183808);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92963870896963584);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92963489215295488);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92963485000015872);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92956157563506688);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92956610548334592);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92952126837370882);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92950201656016896);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92948431647158276);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92941813048025088);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92924738028052480);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92911425323925504);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92911374958723072);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92909898584702976);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92908279587549184);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92902088807419904);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92891158426042368);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92884183315263488);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92884023910744064);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92883017290362880);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92863555715608576);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92861307560267776);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92849207001620480);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92836661846740993);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92840348639965184);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92836619882741760);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92800389505761280);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92786657341882368);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92805955326181376);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92788557361582081);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92799852622262273);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92799869424644096);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92793485664587777);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92811353395429376);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92792495825620992);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92793661837942784);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92804239860056065);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92834921210576896);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92744177418379265);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92706143507460096);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92683586527956992);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92655262384652288);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92665785893392384);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92660501053575168);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92656663290593280);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92665370678276096);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92645766472015872);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92642931143491585);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92641547014770688);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92635146490101760);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92635691753799680);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92629115097710592);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92616012100415488);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92614783152558080);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92610555310903296);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92609959694569472);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92608734978772993);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92599905973059584);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92581773988470785);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92586790380253184);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92575402828115970);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92572697510412288);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92572492002111488);
insert into thesis_2017.bonus_users values ('A7HLNMRH2IKXG', 92560261399068672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90960910386077697);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90960784540180480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90960574837571586);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90960281227886592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90960230887862272);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90957991121133568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90957013865086977);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90954354688921600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90953721344823296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90953733931929601);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90950751773401088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90950084874866688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90949963227471873);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90949799658000384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90949338280370176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90947325039616001);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90933735490465792);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90929201443651585);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90917616784375808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90916912141312000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90915007914704896);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90914651373715457);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90909228138635264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90908036989849602);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90907424604688385);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90904786404249600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90902471119093761);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90895638631428097);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90892958458593281);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90890206990970881);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90888915153719296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90888021762777088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90884364325502976);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90882141323411457);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90878592959000576);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90864848245768192);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90861996081295361);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90861618627477504);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90858946855837697);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90857755648344064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90855541047427072);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90853708174327808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90853582320054273);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90852051394904064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90849262182744065);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90841456587190274);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90840072454287360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90838168257036288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90837203554549760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90835936899899393);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90834745709166592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90830907904241665);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90830656271167488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90823781790134272);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90822280229294082);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90816521437323264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90816366248075265);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90810104173166592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90806685832187904);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90804995527680001);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90483091046535169);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90474845070036992);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90468935304101890);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90465919582736386);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90465412080336896);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90463142940909568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90462341845618688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90448018297470978);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90439847784882177);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90436274254659584);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90432612631449600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90423171253157890);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90422529520443392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90413503361449984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90411150382084097);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 90406247223930880);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 89173012804415489);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 89169074361344000);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 89155287679893504);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 89118990168883200);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 89005420986892288);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88952207843663873);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88855222960656384);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88687727624732673);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88673081115164675);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88668714840506368);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88666101810073600);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88653367873769472);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88643574199095296);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 88641066005315584);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89732595843219456);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89731937362649088);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89729462714904576);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89703504154861568);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89693676934148096);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89683136610443264);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89683006599602176);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89671363232661504);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89636751819280385);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89621954297991168);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89612118688665600);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89569680712409088);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89524696806203392);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89491985429307392);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89354147987591168);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89350373126578176);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89339535057629184);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89338817810661376);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89338310312460289);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89317875667574784);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89305422787379200);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89284518384635904);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89281255199350784);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 89255896441556992);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87141556187963392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87141090616020992);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87140163674845184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87140109148889088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87135906443694080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87131284341669888);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87129124262522880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87126314095616000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87115979330564096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87112841986977792);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87104499503726592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87084333277519872);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87081606984122368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87081099469139968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87072543118327808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87068231352844288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87066444587728897);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87064339038732288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87060471903039488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87045858935316480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87037680025731073);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87035033424117760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87030700737437696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87027441759031297);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87023801103163392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87017786458644480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87011255939891200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 87003563594747905);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86994399006965761);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86992440262791168);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86988560535785475);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86984013935415297);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86982470406373376);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86979354038505472);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86976229311381505);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86972831891595264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86972508934381568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86972043400200193);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86966473356091392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86965894525358080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86965571555565568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86962006413946880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86961905729679360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86961670857039873);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86949146677882880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86945791213703170);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86944327401615360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86943081731067904);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86942054088843264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86939155833171968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86938853872644096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86938480567005184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86936421176311810);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86936114966966272);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86933304779083777);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86928397464387586);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86921577530273793);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86917429346832385);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86912731713777664);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86908994605690880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86906863878279168);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86906540916879360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86902753489723392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86893827994025984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86893760901939201);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86891659547250688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86891567272558592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86890157965459456);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86883421921611776);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86882499174744064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86879110172909568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86878455874076672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86877797380923392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86873393361719296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86872785166680064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86869085811523585);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86864547553615872);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86856733594624000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86852459565289472);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86851050300112896);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86847162192896000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86845979399176192);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86843928384512000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86838559662809088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86835711722000384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86830254919917569);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86828661092782080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86828329751158784);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86824911389212672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86823632118087680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86822847774851075);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86821698552336384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86818888389627904);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86817659425013760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86817298740027393);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86816292090294274);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86814933160951808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86813049889112065);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86812118749429761);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86811267305701376);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86807056245460992);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86804862628675584);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86802450887098368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86802115359547392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86796918587535360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86792426517299200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86790337728749568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86789008159555584);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86785816285806593);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86784411181400064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86783018697637889);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86781982696161280);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86773203992715264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86771660509818880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86768057598484480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86765025116700672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86763422892556288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86759799005519872);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86759215988871168);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86756846223884288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86755701187280896);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86754153468141568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86752425448452096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86751087436115968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86750319903645697);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86744745669427200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86741872571195392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86728236872114176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86725091152498688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86722012520792064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86698847388172288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86698587358109696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86696687321616384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86694741147779072);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86690303595122688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86689301152276480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86687757656793088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86675887759691776);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86656237453836288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86655180497633281);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86653012038254592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86638063526227968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86635211403698176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86633957302599681);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86633605002035201);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86630249546252288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86630257951637504);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86628949324599296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86628806726647809);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86614441214484480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86606115533623296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86572452020371457);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86551115616882688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86542387249299456);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86532568387829760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86518077096869888);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86516906860871680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86510074364829696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86508681826537472);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86508618932944897);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86506551145267201);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86487324430581760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86481867666239488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86480387081125888);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86476968702377984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86474963820888064);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86470224282533888);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86463693717651457);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86456089465470977);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86453916811796480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86449865135095809);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86448191582650370);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86447117840826368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86446287364435968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86445742104915968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86444869706461184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86442021774045185);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86437835846074368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86436049068359680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86434778202652672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86427429777838080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86403740336263168);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86396119302672384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86396027032174592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86395116872404993);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86394059911995394);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86388733145915392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86388473082290176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86387709744132096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86385641927098369);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86384702411382786);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86384220053843969);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 86383687406592000);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89145229705347072);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89137365393752064);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89131740840460288);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89121800323211264);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89116368707915776);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88766018524168192);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88759613821952000);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88709928092565504);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88705998000369664);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88694534963339264);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88684032447098881);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88665577488523268);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88630668321488896);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88624867573903360);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 88614033716031489);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94669090337210368);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94675901899485184);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94703231992729600);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94671065850191872);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94678456234819584);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94685439742574592);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94711108862099456);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94711381508624384);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94650635416387585);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94723968627523584);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94633581363724288);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94631677162295296);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94629445784190977);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94595031507279872);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94585330073739265);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94546197217427456);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94598760268693504);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94584923264000000);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94548562821652480);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94588966552088578);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94493038608523264);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94496054338265088);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94482729042837504);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94496704455389186);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94484490616963072);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94515260060467200);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94495001546985473);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94502802948227072);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94477284819476480);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94469168841228288);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94468833317888000);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94422368818184192);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94410784133742593);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94405151183474688);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94236863115689984);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94192470610558976);
insert into thesis_2017.bonus_users values ('A1BSA7FVVJGZ15', 94191132631773184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95252790665940993);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95222402912493568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95180086608789504);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95177460978696192);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95158523670953984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95133546582253568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95144321774399488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95161925259894785);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95114626081095680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95116941349498880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95125350933209088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95173350539792384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95101325951516673);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95096984855257088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95040915386798081);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94992399859843073);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94989543568191488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95053083062714369);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95033843781869568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 95052869178359808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94980408353103872);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94941682365247488);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94922526949511168);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94963052323155968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94967502488076288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94919855190454274);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94909033890332672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94903199621840897);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94937588699365377);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94931586650345472);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94934099063603201);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94925840462254080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94926721274494976);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94895289156120576);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94859436258107392);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94868630155694080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94834853433970688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94846790414778369);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94819875582775296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94827945415278593);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94880936260403200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94851999757123585);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94870697960144896);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94876788064403456);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94825701458448384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94880302899527681);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94820387262705664);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94816746619404288);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94814192284073984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94813105950949377);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94810962678394880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94811952517365761);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94810308362768384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94809658245660672);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94809540809326592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94809310126817282);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94808529982066688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94808538353897473);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94807846302130176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94807045206851584);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94806629962354688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94803714904301568);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94799570940334082);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94788917416566785);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94769501987549184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94771854979497984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94784119136980992);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94751848149422081);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94746668192378880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94669090337210368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94703231992729600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94685439742574592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94711108862099456);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94682252071546880);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94650635416387585);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94631677162295296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94629445784190977);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94624060314632192);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94595031507279872);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94585330073739265);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94584923264000000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94588966552088578);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94493038608523264);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94496054338265088);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94502442250665984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94524017746247680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94515260060467200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94502802948227072);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94468833317888000);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94463330391031808);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94457542243135489);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94455524766121984);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94449942168485889);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94427448120311809);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94419231457816576);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94440932799299585);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94443382256046080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94422368818184192);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94412973560430592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94410784133742593);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94409056097284096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94432808415674369);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94414718386712576);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94404371055525890);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94405151183474688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94439422858244096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94401304998322176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94400831033585664);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94350059013021696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94325912392306688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94244358357913600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94265208238903296);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94168487580282881);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94160807792885760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94147520233603073);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94143971869204480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94127798620393473);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94128964641095680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94084521816883200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94032239804956673);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94052447974195200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 94022030869012480);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93961259606749185);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93972412248494080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93983954998276096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93985246839709696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93939990299545600);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93889167897010176);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93703104381464576);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93461994820222976);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93388393161031680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93344399106375680);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93309473153761281);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93290426794127360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93212920250511360);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93154434896510976);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93139679330828289);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93143353545342976);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93064341263355904);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93055168320516096);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93026932257587200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 93003712594837505);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92988164297326592);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92967440228691968);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92956157563506688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92816671798083584);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92744177418379265);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92729811943952384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92683091608477696);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 92675508285865985);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 92614783152558080);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 92572492002111488);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 92553877664186368);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 92288789266972672);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 92024581686169600);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91904414859796480);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91859422560792576);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91563669623808001);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91232764178857984);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91141059941371905);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 91074336919785472);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 90954354688921600);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 90953721344823296);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 91203274040029184);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 91174782128754688);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 91170067726872576);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 91126170153795585);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 91563669623808001);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 90221328732405760);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 90215536432128001);
insert into thesis_2017.bonus_users values ('A3UVW94KB9UBAJ', 89863105814872065);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92530704121999360);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92517353669144576);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92511813010325504);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92496772215226368);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92496495412129792);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92456876020736000);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92467420475817985);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92449041031495680);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92437519282618368);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92417495700471808);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92415809581879296);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92389444174364672);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92384360698875904);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92358049821507584);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92333769004036096);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92288789266972672);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92287308698615808);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92268631450329088);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92247915778686976);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92239380349059072);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92227279786221568);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92218358530965504);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92202780877537280);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92161248879321088);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92054461912068096);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92047570645426176);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 91942717256507394);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 103996572328669184);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 103995788006391808);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103256231522734080);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103253358411919360);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103245334746112001);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103232290431311873);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103154955849957376);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103143744475373569);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103067542360309760);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 103061980742565888);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102878278590791680);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102809445897224192);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102726784524623873);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102724301488271360);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102565891039952899);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102564544668372994);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102512925377437696);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92645766472015872);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92641547014770688);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92640443925397504);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92614783152558080);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92608734978772993);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92572492002111488);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92493450326441984);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92247915778686976);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92218836656455680);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 92202780877537280);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 91611178501013505);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 91285872451911680);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 91268315091181568);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 91261851702280192);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 91202682622197760);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 90693234099109888);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 90520256799457280);
insert into thesis_2017.bonus_users values ('A11TREGDHSUSJW', 90104626434670592);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100199108190547970);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100193890467987456);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100129231111061504);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100066303951376384);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100010301621153793);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100005188731011072);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 99893100188676097);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 99471560028602368);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101745497768861697);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101696487318237184);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101675339650052097);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101650727440617473);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101608465629319169);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101423391982096384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101397446013358080);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101390005309681664);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101384749850963969);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101296631726608384);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101230508528242688);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101194454299443200);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101192118076325888);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 101004259365109760);
insert into thesis_2017.bonus_users values ('A5EU1AQJNC7F2', 100996218888531968);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105479334302711808);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105463215554695168);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105438209152004096);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105419674501660674);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105413018162176000);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105385272837025792);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105384446554947584);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105378889102143488);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105378629025939456);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105375835632058368);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105074693018619904);
insert into thesis_2017.bonus_users values ('A2VRDE2FHCBMF8', 105017046479282178);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104728662913466368);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104723269042700288);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104713123008749568);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104707917898452992);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104687403536625664);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104662082523373568);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104641668875173888);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104633246704349184);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104630289699053569);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104617832632958977);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104616326856843264);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104610027024826368);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104575637938913280);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104569581338763264);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104544696549916673);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104537968886300672);
insert into thesis_2017.bonus_users values ('A3PVIRB9JTIZHJ', 104428677890187264);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100195262017974272);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100170033304584192);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100129231111061504);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100128308355796993);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100083060200054784);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100072738030485505);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100066303951376384);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100019285816127488);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100015854879641600);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100005188731011072);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 100004769313210368);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99967121257271296);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99943024964018177);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99919255834865665);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99913866187771904);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99909462147612672);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99906412888592384);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99858480369901568);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99789551207321601);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99786766185279489);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99769913476005888);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99649985745911808);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99639567090589696);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99637759324598272);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99635767051165696);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99575213884321792);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99530271904382976);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99457542689792000);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99369420341784576);
insert into thesis_2017.bonus_users values ('A1YILFU07P6DND', 99367574856417280);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102388446806540289);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102381828211613696);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102076617077170177);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102070858306170881);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102067091837943808);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102064038359482368);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 102040684487393280);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 101983281247043584);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 101947365405106176);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 101816868033134592);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 101766674788786176);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98454114152878081);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98166556860760064);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98160135406493696);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98086206579027968);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98086030418255874);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98079151780675586);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98050357883703296);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 98035291943731202);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97986298270330880);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97957776994742272);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97897324507971584);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97833541735415809);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97830647644684288);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97829502591320067);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 97826855977107456);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103256466412142592);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103256231522734080);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103255166177902593);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103245380845707265);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103230604333686784);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103228515574497280);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103227525722943488);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103218717676142594);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103198337536233474);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103161213776695296);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103159624114515968);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103157208207990785);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103143744475373569);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 103060877640601600);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102899136889765888);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102895416512749569);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102893252256075778);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102893046730993665);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102866861716283393);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102844237619539968);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102839430968127490);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102809445897224192);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102785500582129667);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102753510633914368);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102750293602738176);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102743314272497664);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102739656864571392);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102727367532888064);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102719536775696385);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102695285314174976);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102632031019732993);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102568306963259392);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102565891039952899);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102564544668372994);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102557082988969984);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102555749191925760);
insert into thesis_2017.bonus_users values ('A25CAT0W9W97Z8', 102552670585356289);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100965369782599680);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100926832542613504);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100920012604309504);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100907719086718976);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100895593358032896);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100870796616024064);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100853629321355264);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100842208256737281);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100822536954183681);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100741842760515584);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100687606215479296);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100686318538989570);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100684418519281664);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100636364374163457);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100638050509529089);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100618471477485568);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100573521150873600);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100282667127406592);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100281454973562880);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100278435083075584);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100214413197459456);
insert into thesis_2017.bonus_users values ('AU7A3QNJF3O00', 100205718443008001);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100991357706977280);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100965369782599680);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100964442870788098);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100961250967687169);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100960122716700672);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100941969785765888);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100939872616976384);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100933023326932993);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100932914283413505);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100895593358032896);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100890220450414592);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100879898255695872);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100879155889053696);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100869823537487872);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100853629321355264);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100777699869597696);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100686318538989570);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100684418519281664);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 100343258055581696);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95252790665940993);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95236961375232000);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95199325847699456);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95222402912493568);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95235770159337472);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95247862341971968);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95177460978696192);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95177448383197184);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95176332698320896);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95175376376041472);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95114714169884672);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95159228322430976);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95114026316599296);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95144321774399488);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95161925259894785);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95163854656520193);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95116941349498880);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95117293679411200);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95121748017684480);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95116324774227969);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95091507102629888);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94992399859843073);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95053083062714369);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 95087757394845696);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94607182431141888);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94311920198356992);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94117476471803904);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94127798620393473);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94129014989524992);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94091039769493504);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94084521816883200);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94084240769155072);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94081891975704576);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 94052447974195200);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93725371933007874);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93681201713385472);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93638885397106688);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93602076176822272);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93471264223666176);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93442650690166787);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93434677318254593);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93431858724995073);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93228338537177088);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93152702636359680);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93174236205686784);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93199804670296065);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93077075140943872);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 93067596022284288);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92863555715608576);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92861307560267776);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92816671798083584);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92757339169497088);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92733591003480064);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92696039429115904);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92682965779353600);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92676447835131904);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92675508285865985);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92672589058682880);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92672823922929664);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92655841194426369);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92655262384652288);
insert into thesis_2017.bonus_users values ('AEI5USUUPBU2G', 92662665335406592);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 104575637938913280);
insert into thesis_2017.bonus_users values ('A2DWQF4ZHHBLOW', 104102008734224384);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92645766472015872);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92646647275847680);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92645783245033472);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92655325282447360);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92641547014770688);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92640443925397504);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92639365972500480);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92639047213780992);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92637960876462080);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92634160853819392);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92625449288601600);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92610555310903296);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92609959694569472);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92572492002111488);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92503592145125376);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92496495412129792);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92437519282618368);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92403180524154881);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92288789266972672);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92280706843156481);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92275757551857666);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92243130073624576);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92231037882605568);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92202587931160576);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92054461912068096);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 92024581686169600);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91944394965524480);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91917941511176192);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91904414859796480);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91884470927491073);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91860911538708480);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91859422560792576);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91813843075993600);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91742648943128576);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91741533270847488);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91680942325899265);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91585215755063296);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91448837968896000);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91363966227447808);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91206046487543809);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91202913325694976);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91202716205981696);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91194650546806784);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 91133103325708288);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90967176659480577);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90970859275169792);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90909182013882368);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90833781019258881);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90833042830147584);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90785240355840000);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90520256799457280);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90461863686586368);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90432612631449600);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90234767307575296);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90224302519095296);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90221223883190272);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 90078823084867584);
insert into thesis_2017.bonus_users values ('A1MJVTR0PCKBWW', 89861990138388480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101750035984818176);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101746961551593472);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101745497768861697);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101745384493293568);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101743698399870976);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101743505478664192);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101742830157955072);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101739051127816193);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101735125259272192);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101732717724573696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101725927121235970);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101722106110279680);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101720088641679360);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101719967027830786);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101714392776843265);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101707702899712002);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101696487318237184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101691873562869761);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101679433265586177);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101678669919035392);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101674140070711297);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101672542032502784);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101671409570426880);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101656855339737089);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101655068566237184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101653000757579776);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101650727440617473);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101644251468808192);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101636252922683393);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101635695063465984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101635556659822592);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101626073321717760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101625993646706688);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101622336209428480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101617596629139456);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101608465629319169);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101595111003127808);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101592929952468992);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101587791938453504);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101557139935473664);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101553008567005184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101514160902389760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101486470128336896);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101476131177365505);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101469780975951873);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101457277768314880);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101457051267497984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101451435094441984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101450956939599872);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101447647629545472);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101435672887443456);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101435391873265664);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101425103258128384);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101423391982096384);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101417549312425984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101417314423029761);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101414588129624064);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101410519646351360);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101408347013652480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101401124409589761);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101397446013358080);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101385475478142976);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101384758264741888);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101377061716897794);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101359403671879680);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101357965038206978);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101357193265291264);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101349442195689472);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101346929828573185);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101344618758672385);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101330794345275392);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101326969119047680);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101322133107523585);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101320396640493568);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101317452251676673);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101315086651637760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101314524614885377);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101298724667523074);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101296631726608384);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101291946680664064);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101290084405493761);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101284086550773760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101280961823653888);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101280953439236096);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101259096887537664);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101259059134611457);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101258312548491264);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101249319969099776);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101245360546136064);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101242973995532288);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101239127806181376);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101231641007104002);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101230508528242688);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101225873839112194);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101221968942075904);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101208656212795392);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101194454299443200);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101172148973993984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101172035752964096);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101151815000793088);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101147016691843073);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101135050367700992);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101113214795915264);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101112589848821760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101105199493554176);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101077500297359361);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101071263400865793);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101056528781557760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101052082823507970);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101051873133469696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101046936404103168);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101039441203830785);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101036182242205698);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101026904408199168);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101019023323574273);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101018343833735168);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101017794409267200);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101015458186145792);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101013830783610880);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101013423919349761);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101010068509696000);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101009732940206080);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101008734687465473);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101006792728920065);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101004259365109760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101001080116215809);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101000560014139392);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 100999293342724097);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 100997460415090688);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 100996218888531968);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 100993182208245760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102506470318415872);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102503584649846784);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102500996743315456);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102499369361752065);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102493157626880000);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102492247458717696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102485976970047488);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102482990621392896);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102470898426187776);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102469610770661376);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102464409833713664);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102463935906717696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102459796103507968);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102437750862659584);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102433514611425280);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102431643964416001);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102428707951607810);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102422785569198080);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102421342745399296);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102420369662685184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102418297680691200);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102410341060841473);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102399855305035776);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102396646670864384);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102393727443664896);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102381828211613696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102381752688979968);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102380729291382784);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102374895014514689);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102364937741021184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102363348074631168);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102361750044803072);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102348064051834880);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102341344755855362);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102337217590079488);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102330456359444480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102288219705585665);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102252232585641984);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102251267908321280);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102220146164244480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102219558965874688);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102213590467092480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102177762713931777);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102176659641352192);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102171739685007360);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102163623727730688);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102159731396837376);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102151250514153472);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102133290537975809);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102103653564952576);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102100654620803072);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102090290499817472);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102089803994116096);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102078487745142784);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102076617077170177);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102070858306170881);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102067213443403776);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102067091837943808);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102064097104896000);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102064038359482368);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102055423271645185);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102044278989127681);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102043016495247361);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102017930367221760);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 102008988111089664);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101994635236356096);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101994559730483201);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101988658332176384);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101988528321335297);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101983281247043584);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101978503909613568);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101970039804141568);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101966936052740097);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101966487237050368);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101927845131071488);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101895209251651584);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101873705059229696);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101861365404286976);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101858425205571584);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101841127883288576);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101833964028821506);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101829648081629184);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101827999711764480);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101822274495193088);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101816868033134592);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101804138299523072);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101799822377500673);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101795577741852672);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101793287647666176);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101792989860470784);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101791874192388097);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101781426176933888);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101769468220424192);
insert into thesis_2017.bonus_users values ('A2W3QPD493MHEI', 101766674788786176);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95252790665940993);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95196851216728065);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95215465533685761);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95229789081841664);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95244691439751168);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95191562190983169);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95228946035118080);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95199325847699456);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95237355631423488);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95190375232307200);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95177460978696192);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95159228322430976);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95158523670953984);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95133546582253568);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95114626081095680);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95115943105138688);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95173350539792384);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95101325951516673);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95096984855257088);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94987433833275392);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94992399859843073);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94989543568191488);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95053083062714369);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94989962986000388);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95059835896344576);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95087757394845696);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 95006723408007170);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94982144794959873);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94980408353103872);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94963052323155968);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94967502488076288);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94919855190454274);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94903199621840897);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94895289156120576);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94868630155694080);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94834853433970688);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94851999757123585);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94816666910851072);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94816746619404288);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94813105950949377);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94808999727341568);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94807846302130176);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94799570940334082);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94769501987549184);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94771854979497984);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94751848149422081);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94746668192378880);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94703231992729600);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94678456234819584);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94711108862099456);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94682252071546880);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94711381508624384);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94714841809420289);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94650635416387585);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94631677162295296);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94629445784190977);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94595031507279872);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94584923264000000);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94548562821652480);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94588966552088578);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94496054338265088);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94502442250665984);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94524017746247680);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94496704455389186);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94515260060467200);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94495001546985473);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94502802948227072);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94463330391031808);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94443382256046080);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94422368818184192);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94412973560430592);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94410784133742593);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94438466531766272);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94402831733362688);
insert into thesis_2017.bonus_users values ('A1ZFUBB7KVAT4G', 94405151183474688);

