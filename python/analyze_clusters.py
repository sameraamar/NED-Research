import re
from pprint import pprint
import pandas
from tweets_stats import hist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np


try:
    from pandasql import sqldf #, load_meat, load_births
except:
    print("try to install using:\npip install pandasql")
    exit()



def loadPositiveLabeled(labeled_file, sep):
    labeled_file = "c:/data/events_db/petrovic/relevance_judgments_00000000 - flat.csv"

    #%% Load judgment info

    labeled_ds = pandas.read_csv(labeled_file, sep=sep) #, header=["id", "topic_id", "status"])

    labeled_ds['id'] = labeled_ds['id'].astype(str)
    labeled_ds['topic_id'] = labeled_ds['topic_id'].astype(int)
    labeled_ds['status'] = labeled_ds['status'].astype(str)

    #print('labeled_ds', labeled_ds.info())

    print("labeled dataset: " , len(labeled_ds))
    print(labeled_ds.head())

    return labeled_ds

#%% Load dataset
def loadThreads(filename, max_clusters=None):

    file = open(filename)

    clusters = []
    members = []

    seen1 = {}
    seen2 = {}

    count = 0
    #members = []
    for line in file:
        line = line.strip()
        count+=1

        match = re.search(r"LEAD: (\d+) SIZE: (\d+) Entropy: (\d+.\d+) Age: (\d+) \(s\)", line)
        if match:
            lead = match.group(1)
            size = match.group(2)
            entropy = match.group(3)
            age = match.group(4)
            if not seen1.get(lead, False):
                clusters.append([lead, int(size), float(entropy), int(age)])
                seen1[lead] = True

        #match = re.search(r"(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)
        match = re.search(r"(\d+)	[^	]+	(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)
        if not match:
            match = re.search(r"(\d+)	(null|\d+)	([-]?\d+.\d+)	", line)

        if match:
            member = match.group(1)
            if not seen2.get(member, False):
                members.append((lead, member))
                seen2[member] = True

            if count%10000 == 0:
                print('lines:', count, '. clusters:', len(clusters), 'documents:', len(members))

        if max_clusters is not None and len(clusters) > max_clusters:
            break

    file.close()

    members  = pandas.DataFrame(members , columns=["lead", "member"])
    members['lead'] = members['lead'].astype(str)
    members['member'] = members['member'].astype(str)

    members = sqldf("SELECT DISTINCT * FROM members ORDER BY lead, member;", locals())

    #print('members', members.info())


    types = [('lead', str), ('size', int), ('entropy', float), ('age', int)]
    clusters = pandas.DataFrame(clusters, columns=["lead", "size", "entropy", "age"])
    for f, t in types:
        clusters[f] = clusters[f].astype(t)
    clusters = sqldf("SELECT DISTINCT * FROM clusters ORDER BY lead;", locals())

    #print('clusters', clusters.info())
    #print('members', members.info())

    clusters.to_csv("c:/temp/clusters.csv", index_label="#")
    members.to_csv("c:/temp/members.csv", index_label="#")

    return clusters, members


def analyze(clusters, members, labeled_ds, entropy, size, minId, maxId, save_out=False):
    #results = sqldf("SELECT COUNT(*) FROM clusters JOIN members ON clusters.lead = members.lead;", locals())
    #print('all clusters: ', results)

    entropy = str(entropy)
    size = str(size)

    #print('before filtering clusters', len(clusters))
    filtered_clusters = sqldf("SELECT * FROM clusters WHERE entropy >= " + entropy + " AND size >= " + size + ";", locals())
    #print('after filtering clusters', len(filtered_clusters))


    members = sqldf("SELECT members.* "
                    "FROM members "
                    "JOIN filtered_clusters "
                    "ON filtered_clusters.lead = members.lead;", locals())



    if save_out:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #ax0 = axes.flatten()
        hist(filtered_clusters['size'], ax, 'filtered clusters sizes', 'sizes', 'count', bins=10, color="blue")
        plt.savefig('c:/temp/clusters_'+ entropy +'_' + size + '.png')
        plt.close()

    #print('calculate true positive... step 1')

    true_positive = sqldf("SELECT members.*, labeled_ds.topic_id, labeled_ds.status "
                          "FROM members "
                          "JOIN labeled_ds "
                          "ON members.member = labeled_ds.id;", locals())

    #print('calculate true positive... step 2')
    true_positive_final = sqldf("SELECT filtered_clusters.*, true_positive.member AS tp "
                    "FROM true_positive "
                    "JOIN filtered_clusters "
                    "ON filtered_clusters.lead = true_positive.lead "
                    "ORDER BY filtered_clusters.lead, true_positive.member;", locals())

    if save_out:
        true_positive.to_csv('c:/temp/true_positive_all_' + entropy + '_' + size + '.csv', index_label="#")
        true_positive_final.to_csv('c:/temp/true_positive_final_' + entropy + '_' + size + '.csv', index_label="#")

    tp = len(true_positive_final)
    print('** Selected Elements (TP + FP): ', selected_elements)
    print('** True Positive (TP): ', tp)
    print('** Relevant Elements (TP + FN): ', len(members))

    precision = 1.0 * tp / selected_elements
    if len(members) == 0:
        recall = ''
    else:
        recall = 1.0 * tp / len(members)

    print('** Precision: [ TP / (TP + FP) ]: ', precision)
    print('** Recall:    [ TP / (TP + FN) ]: ', recall)

    return precision, recall, selected_elements, tp, len(members)

#print('looking for known (labeled) tweets ...')
#for tid in labeled:
#    #print (tid['id'])
#    tmp = df [ df["member"] == tid['id'] ]
#    if(len(tmp) > 0):
#        print ( tmp )


### MAIN

def plotPercision(dataset):
    # set up a figure twice as wide as it is tall
    fig = plt.figure() #figsize=plt.figaspect(0.5))
    fig.suptitle('Precision vs. <size, entropy>')

    # plot a 3D surface like in the example mplot3d/surface3d_demo
    X = dataset['size'].as_matrix()
    Y = dataset['entropy'].as_matrix()
    Z = dataset['precision'].as_matrix()

    print(min(X), max(X))
    print(min(Y), max(Y))
    print(min(Z), max(Z))

    maxZ = max(Z)
    for i in range(len(Z)):
        if Z[i] == maxZ:
            break

    print('best size=', X[i], 'entropy=', Y[i])

    # set up the axes for the second plot
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlabel('size')
    ax.set_ylabel('entropy')
    ax.set_zlabel('precision')

    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    ax.scatter(X, Y, Z, c='r', marker='o')

    plt.show()


if __name__ == "__main__":
    sep = ","
    VERSION = "V1"
    SUFFEX = "15m"
    FOLDER = "c:/temp"



    filename = FOLDER + "/" + "threads_" + SUFFEX + "_" + VERSION + ".txt"
    filename = FOLDER + "/" + "dataset_300k_V1.txt"
    filename = FOLDER + "/threads_big.txt"

    labeled_ds = loadPositiveLabeled("c:/data/events_db/petrovic/relevance_judgments_00000000 - flat.csv", sep)
    clusters, members = loadThreads(filename, max_clusters=None)

    print('all clusters: ', len(clusters))
    print('documents: ', len(members))

    minmax = sqldf("SELECT MIN(member) AS minId, MAX(member) AS maxId FROM members;", locals())
    print('Min / Max document id: ', minmax)

    print('all positive labeled: ', len(labeled_ds))

    minId = minmax['minId'][0]
    maxId = minmax['maxId'][0]
    labeled_ds = sqldf("SELECT * FROM labeled_ds WHERE status = 'Loaded' "
                       "AND id >= '" + minId + "' AND id <= '" + maxId + "';", locals())

    selected_elements = len(labeled_ds)
    print('selected positive labeled: ', len(labeled_ds))

    print('>>>,entropy,size,precision,recall,selected_elements,tp,relevant_elements')
    performance = []
    file = open('c:/temp/performance.csv', 'w', 10)
    file.write('entropy,size,precision,recall,selected_elements,tp,relevant_elements\n')
    for entropy in range(35):
        entropy = entropy / 10.0
        size = 1
        for n in range(9):
            size = size * 2
            results = analyze(clusters, members, labeled_ds, entropy, size, minId=minId, maxId=maxId, save_out=False)

            precision, recall, selected_elements, tp, relevant_elements = results

            line = '{0},{1},{2},{3},{4},{5},{6}\n'.format(entropy,size,precision,recall,selected_elements,tp,relevant_elements)
            print('>>>', line)
            file.write(line)

            #size, precision, recall, selected_elements, tp, relevant_elements)

            performance.append((entropy, size, precision, recall, selected_elements, tp, relevant_elements))

    file.close()
    performance = pandas.DataFrame(performance, columns=['entropy', 'size', 'precision', 'recall', 'selected_elements', 'tp', 'relevant_elements'])

    print(performance.info())

    plotPercision(performance)
