import os
import warnings
from pprint import pprint
import datetime
import numpy as np
import pandas

warnings.filterwarnings("ignore")

try:
    from pandasql import sqldf #, load_meat, load_births
except:
    print("try to install using:\npip install pandasql")
    exit()

try:
    import powerlaw
except:
    print("try to install using:\npip install powerlaw")
    exit()


VERSION = 'V1'
SUFFEX = "5m"
FOLDER = 'C:/data/Thesis/threads_petrovic_all/analysis_3m'

def groupFeatures(dataset, filename):
    print("query feature set...")
    components = sqldf("SELECT root, "
                       "COUNT(*) size, "
                       "COUNT(DISTINCT userId) AS users_count, "
                       "SUM(retweets)/COUNT(*) as retweet_per_tweet, "
                       "MIN(retweets) as retweet_min, "
                       "MAX(retweets) as retweet_max, "
                       "AVG(likes) as likes_avg, "
                       "MAX(likes) as likes_max, "
                       'MAX(depth) as depth_max, '
                       'MAX(topic_id) AS topic_id, '
                       'MAX(is_topic) AS is_event, '
                       'SUM(CASE WHEN parentType=2 THEN 1 ELSE 0 END) AS internal_rtwts, '
                       'SUM(CASE WHEN parentType=1 THEN 1 ELSE 0 END) AS internal_rplys '
                       "FROM dataset "
                       "GROUP BY root;", locals())
    print("writing features to csv file...")
    components.to_csv(filename, sep=",")

    print("done 'groupFeatures'")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (end) group level features %%%%%%%%%%%%%%%%%%%%%%%%

def analyzeDataset(filename, sep=',', sample_size=None):
    np.seterr(divide='ignore', invalid='ignore')

    print('loading ', filename)
    dataset = pandas.read_csv(filename, sep=sep, nrows=sample_size)

    if sample_size is None:
        sample_size = dataset.size

    if False:
        print('identify topics...')

        events_no = sqldf("SELECT * FROM dataset WHERE is_topic == 'no';", locals())

        #events_yes = dataset[dataset['topic_id'] > -1]
        events_yes = sqldf("SELECT * FROM dataset WHERE is_topic == 'yes';", locals())

        print("summary: ")

        groups2 = sqldf("SELECT topic_id, COUNT(*) AS count, MAX(depth) AS maxDepth FROM dataset GROUP BY topic_id;", locals())
        print("count per topic:", groups2)

        print("found", len(events_yes), "tweets with", len(groups2)-1, "events")
        print("found", len(events_no), "tweets without any specific events")

    #"/features_" + SUFFEX + "_" + VERSION + ".csv"
    groupFeatures(dataset, filename + "_features.csv" )

    return sample_size

if __name__ == "__main__":
    start = datetime.datetime.now()
    print(start)

    #analyzeDataset(FOLDER + '/dataset_full_' + SUFFEX + "_" + VERSION + '.txt', sep=',')

    size = analyzeDataset(FOLDER + '/events_yes.txt', sep=',')
    print('processed: ', size, ' rows')

    size = analyzeDataset(FOLDER + '/events_no.txt', sep=',', sample_size=10000000)
    print('processed: ', size, ' rows')

    end = datetime.datetime.now()

    print(end)
    print(end-start)
