SHOW VARIABLES LIKE 'secure_file_priv';

DROP TABLE IF EXISTS thesis_analysis.tree_features ;

CREATE TABLE thesis_analysis.tree_features AS
(SELECT 'tweet_id', 'root_id', 'size', 'is_event')
UNION
    (SELECT 
        c.tweet_id, entropy, users, size, 1 AS is_event
    FROM
        thesis_2017.clusters AS c
    JOIN thesis_2017.tweet2topic AS tt ON c.lead_Id = tt.tweet_id
    LIMIT 10000) 
UNION (SELECT 
    c.tweet_id, entropy, users, size, 0 AS is_event
FROM
    thesis_2017.clusters AS c
        LEFT OUTER JOIN
    thesis_2017.tweet2topic AS tt ON c.lead_Id = tt.tweet_id
LIMIT 10000) 
;


SELECT 
    *
FROM
    thesis_analysis.tree_features 
    INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/features.csv' 
    FIELDS TERMINATED BY ',' 
    LINES TERMINATED BY '\n';

DROP TABLE thesis_analysis.tree_features;

#SELECT * FROM thesis_2017.tweet2topic tt
#WHERE tt.tweet_id =  86383699968528385;