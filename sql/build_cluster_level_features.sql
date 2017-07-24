SHOW VARIABLES LIKE 'secure_file_priv';

DROP TABLE IF EXISTS thesis_analysis.clusters_features01 ;
DROP TABLE IF EXISTS thesis_analysis.clusters_features02 ;
DROP TABLE IF EXISTS thesis_analysis.clusters_features03 ;

CREATE TABLE thesis_analysis.clusters_features01 AS 
SELECT 
        c.lead_id, MAX(size) as size, MAX(entropy) AS entropy, 
        MAX(users) users, MIN(timestamp) AS starttime, SUM(c.timestamp) AS sum_timestamp,
        (SUM(c.timestamp) / MAX(size) - MIN(timestamp)) as avg_time 
    FROM
        thesis_2017.clusters AS c 
        GROUP BY c.lead_Id;
        


#CREATE INDEX clusters_features01_idx
#ON thesis_analysis.clusters_features01  (lead_id);


#CREATE TABLE thesis_analysis.clusters_features02 AS
#SELECT e.*, f.size, f.entropy, f.users, (f.sum_timestamp / f.size - e.starttime) as avg_time 
#FROM thesis_analysis.clusters_features01 AS f
#JOIN thesis_2017.clusters e
#        ON f.lead_Id = e.lead_id;

#####################################################

CREATE TABLE thesis_analysis.clusters_features03 AS
SELECT lead_id, entropy, users, size, avg_time, 
IF(tt.tweet_id IS NULL, 0, 1) AS is_leader_event
FROM
    thesis_analysis.clusters_features01 c
        LEFT JOIN
    thesis_2017.tweet2topic tt ON c.lead_Id = tt.tweet_id; 


CREATE TABLE thesis_analysis.clusters_features02 AS
(SELECT 'lead_id', 'entropy', 'users', 'size', 'avg_time', 'is_leader_event')
UNION
(SELECT lead_id, entropy, users, size, avg_time, is_leader_event
FROM    thesis_analysis.clusters_features03  );



SELECT 
    *
    FROM
    thesis_analysis.clusters_features02 
    INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/cluster_features.csv' 
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';



#UNION
#    (SELECT 
#        lead_id, entropy, users, size, starttime, 1 AS is_event
#    FROM
#        thesis_analysis.clusters_features02 AS c
#    INNER JOIN thesis_2017.tweet2topic AS tt ON c.lead_Id = tt.tweet_id
#    #LIMIT 10000
#    ) 
#UNION (SELECT 
#    lead_id, entropy, users, size, starttime, 0 AS is_event
#FROM
#    thesis_analysis.clusters_features02 AS c
#         LEFT JOIN
#    thesis_2017.tweet2topic tt ON c.lead_Id = tt.tweet_id 
#    WHERE tt.tweet_id IS NULL
##LIMIT 10000
#) ;
