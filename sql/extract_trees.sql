SELECT i.*, tt.topic_id FROM thesis_2017.id2tree_vw i
join thesis_2017.dataset 
left outer join thesis_2017.tweet2topic tt ON tt.tweet_id= i.tweet_id
where i.root_id is  not null and i.tree_size>10  
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/trees.csv' 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n';

