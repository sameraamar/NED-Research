SHOW VARIABLES LIKE 'secure_file_priv';

DROP TABLE IF EXISTS thesis_analysis.tree_features00 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features01 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features02 ;
DROP TABLE IF EXISTS thesis_analysis.tree_features03 ;

CREATE TABLE thesis_analysis.tree_features00 AS SELECT c.*, 
IF(tt.tweet_id IS NULL, 0, 1) AS is_tweet_event, d.likes, d.retweets, d.user 
 FROM
    thesis_2017.id2tree c
    JOIN
    thesis_2017.dataset d ON c.tweet_id = d.tweet_id
        LEFT OUTER JOIN
    thesis_2017.tweet2topic_vw tt ON c.tweet_id = tt.tweet_id
    ;

CREATE TABLE thesis_analysis.tree_features01 AS SELECT c.root_id,
    COUNT(c.tweet_id) AS tree_size,
    SUM(IF(parentType = 'rtwt' OR parentType = 'qte', 1, 0)) AS rtwt_count,
    SUM(IF(parentType = 'rtwt', 1, 0)) AS rtwt_only_count,
    SUM(IF(parentType = 'rply', 1, 0)) AS rply_count,
    SUM(IF(parentType = 'qte', 1, 0)) AS qwt_only_count,
    SUM(IF(parentType = '', 1, 0)) AS original_count,
    IF(SUM(is_tweet_event) > 0, 1, 0) AS is_tree_event ,
    AVG(likes) AS likes_avg,
    AVG(retweets) AS retweets_avg
    FROM
    thesis_analysis.tree_features00 c
GROUP BY c.root_id;



CREATE TABLE thesis_analysis.tree_features02 AS (SELECT 'tweet_id',
    'parent',
    'parentType',
    'root_id',
    'depth',
    'timestamp',
    'time_delta',
    'likes', 
    'retweets', 
    'user',
    'is_tweet_event') UNION (SELECT 
    tweet_id,
    parent,
    parentType,
    root_id,
    depth,
    timestamp,
    time_delta,
    likes, 
    retweets, 
    user,
    is_tweet_event
FROM
    thesis_analysis.tree_features00);



SELECT 
    *
FROM
    thesis_analysis.tree_features02 INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tree_tweet_features.csv' FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n';



CREATE TABLE thesis_analysis.tree_features03 AS (SELECT 'root_id',
    'tree_size',
    'original_count',
    'rtwt_count',
    'rtwt_only_count',
    'rply_count',
    'qwt_only_count',
    'likes_avg',
    'retweets_avg',
    'is_tree_event') UNION (SELECT 
    root_id,
    tree_size,
    original_count,
    rtwt_count,
    rtwt_only_count,
    rply_count,
    qwt_only_count,
    likes_avg,
    retweets_avg,
    is_tree_event
FROM
    thesis_analysis.tree_features01);



SELECT 
    *
FROM
    thesis_analysis.tree_features03 INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tree_features.csv' FIELDS TERMINATED BY ',' LINES TERMINATED BY '
';

