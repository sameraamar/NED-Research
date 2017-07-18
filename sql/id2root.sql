DROP TABLE IF EXISTS thesis_analysis.id2root_level0;
CREATE TABLE thesis_analysis.id2root_level0 AS
SELECT d1.tweet_id, d1.parent, d1.parentType, 0 AS depth
    FROM
        `thesis_2017`.`dataset` d1
        WHERE d1.parent IS NULL;
        
DROP TABLE IF EXISTS thesis_analysis.id2root_level1;
CREATE TABLE thesis_analysis.id2root_level1 AS
SELECT d1.tweet_id, d2.parent parent1, d2.parentType parent1Type
    FROM
        (`thesis_2017`.`dataset` d1
        JOIN `thesis_2017`.`dataset` d2 ON (d1.parent = d2.`tweet_id`));
        

DROP TABLE IF EXISTS thesis_analysis.id2root_level2;
CREATE TABLE thesis_analysis.id2root_level2 AS
SELECT d1.tweet_id, d1.parent parent1, d1.parentType paren1Type, 
	   d2.parent parent2, d2.parentType parent2Type, d3.tweet_id parent3
    FROM
        (`thesis_2017`.`dataset` d1
        JOIN `thesis_2017`.`dataset` d2 ON (d1.parent = d2.`tweet_id`)
        JOIN `thesis_2017`.`dataset` d3 ON (d2.parent = d3.`tweet_id`));
        
        
DROP TABLE IF EXISTS thesis_analysis.id2root_level3;
CREATE TABLE thesis_analysis.id2root_level3 AS
SELECT d1.tweet_id, d1.parent parent1, d1.parentType paren1Type, 
	   d2.parent parent2, d2.parentType parent2Type, 
       d3.parent parent3, d3.parentType parent3Type, 
       d4.tweet_id parent4
    FROM
        (`thesis_2017`.`dataset` d1
        JOIN `thesis_2017`.`dataset` d2 ON (d1.parent = d2.`tweet_id`)
        JOIN `thesis_2017`.`dataset` d3 ON (d2.parent = d3.`tweet_id`)
        JOIN `thesis_2017`.`dataset` d4 ON (d3.parent = d4.`tweet_id`));
        
        
        