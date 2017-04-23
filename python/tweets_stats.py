from pprint import pprint
import pandas
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
import matplotlib.mlab as mlab
import os
import warnings
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
SUFFEX = "15m"
CUT_FROM=None
img_idx = 0
SHOW_ONLY=False
GENERATE_GRAPH=True
FOLDER = 'c:/temp/15m_'

def savefig(fig):
    newname = FOLDER + '/img_' + str(CUT_FROM) + '_' + SUFFEX + "_" + VERSION + "_" + str(img_idx) + ".png"
    if os.path.exists(newname):
        os.remove(newname)
        print('removed: ' + newname)

    print('save to file: ' + newname)
    fig.savefig(newname)
    print('done.')

# find the first bin for the given binsize
def bin_find_k(binsize):
    print
    'calculate bin for binsize = ', binsize
    k = 1
    flag = True
    while flag:
        logk = math.log10(k)
        logk1 = math.log10(k + 1)

        print
        k, '\t', logk, '\t', logk1, '\t', abs(logk1 - logk)
        if abs(logk1 - logk) <= binsize:
            flag = False

        k += 1

    return k


def findBin(val, bins):
    for j in range(len(bins) + 1):
        # print i, j, logi >= bins[j] , logi < bins[j+1]
        if val >= bins[j] and val < bins[j + 1]:
            return j
    return -1

def log_hist(arr, maxBinSize):
    K = bin_find_k(maxBinSize)

    #logY = np.log10(arr)
    size = len(arr)
    logX = np.log10(range(size))

    # calculate bin array
    print
    "Min and max:"
    minV = logX[K]  # np.min(logX[k:])
    maxV = logX[len(logX) - 1]  # np.max(logX[k:])

    print
    minV, maxV
    # create log bins: (by specifying the multiplier)
    bins = [minV]
    cur_value = bins[0]
    while cur_value <= maxV:
        cur_value = cur_value + maxBinSize
        bins.append(cur_value)

    bin_indices = {}
    for i in range(K, len(arr)):
        j = findBin(np.log10(i), bins)
        if j not in bin_indices:
            bin_indices[j] = []
        bin_indices[j].append(i)

    # %%

    newarr = arr[:]  # copy values to a new array
    average = [0] * len(bins)
    currbin = K
    for b in bin_indices:
        for ind in bin_indices[b]:
            average[b] += newarr.iloc[ind]
        average[b] = average[b] / len(bin_indices[b])  # average

        for ind in bin_indices[b]:
            newarr[ind] = average[b]

    return bins, average

def np_hist(actions, title, ax, color, log=False, bins=100, normalize=False):
    hist, edges = np.histogram(actions, bins=bins, density=normalize)
    if normalize:
        hist = hist / len(actions)
    #if(normed):
    #    weights = np.ones_like(feature) / len(feature)
    #    ax.hist(feature, bins=100, weights=weights, histtype='bar', color="red")

    colors = [color]

    ax.bar(edges[:-1], hist, width=1, color=color, label=title)
    ax.legend(prop={'size': 10})
    ax.set_title('bars with legend')


def hist(feature, ax, title, xlabel, ylabel, bins=100, normed=0, log=False, color="blue", cut_from=None):
    #np_hist(feature, title, ax, color, log=log, bins=bins, normalize=(normed == 1))

    #feature = feature[feature > 0]

    log = False
    if log:
        bins, average = log_hist(feature, 0.01)
        ax.plot(bins, average)
        #plt.show()
    else:
        hist, xbins = np.histogram(feature, bins=bins)
        #xbins = np.linspace(0, )
        if(cut_from is not None and cut_from>0):
            cut_from = 0
            sum_all = len(feature)
            too_high = (hist[cut_from] / sum_all) > 0.95
            while(too_high):
                sum_all -= hist[cut_from]
                cut_from+=1
                too_high = (hist[cut_from] / sum_all) > 0.95

            hist, xbins = hist[cut_from:], xbins[cut_from:]
        elif(cut_from is not None and cut_from<0):
            ordinal_bin_k = -1*cut_from
            newhist = []
            newxbin = []

        print("maximum: ", max(hist))

        tmp1 = [np.sum(xbins[0:9])]
        center = (xbins[:-1] + xbins[1:]) / 2
        width = np.diff(xbins)

        ax.bar(center, hist, align='center', width=width, color=color)
        #ax.set_xticks(xbins)

        #ax.hist(bins, histtype='bar', normed=normed, log=log, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

def hist1(feature, ax, title, xlabel, ylabel, bins=100, normed=0, log=False, color="blue"):
    # np_hist(feature, title, ax, color, log=log, bins=bins, normalize=(normed == 1))

    #feature = feature[feature > 0]

    #if log:
    #    bins, average = log_hist(feature, 0.1)
    #    ax.plot(bins, average)
    #    # plt.show()
    #else:
    ax.hist(feature, bins=bins, histtype='bar', normed=normed, log=log, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


#sample histogram: http://matplotlib.org/examples/statistics/histogram_demo_multihist.html

def retweets(events_yes, events_no):

    feature_events_no = events_no['retweets']
    feature_events = events_yes['retweets']

    feature_hist(feature_events, feature_events_no, 'retweets', cut_from=CUT_FROM)

    #analyze(feature_events_no, feature_name='retweets-no events_yes')
    #analyze(feature_events, feature_name='retweets-events_yes')

def analyze(data, feature_name, ax1, ax2):
    print("Analyzing feature:", feature_name)
    results = powerlaw.Fit(data)
    print ("alpha=", results.power_law.alpha)
    print("segma=", results.power_law.sigma)
    print("xmin=", results.power_law.xmin)
    R, p = results.distribution_compare('power_law', 'lognormal')

    print("R=", R, "p=", p)

    ax1.set_title(feature_name)
    ax1.set_xlabel("count")
    #ax.set_ylabel("CCDF")

    results.plot_ccdf(ax=ax1, label="ccdf")

    #ax2.set_title(feature_name)
    #ax2.set_xlabel("count")
    #ax2.set_ylabel("CDF")

    results.plot_cdf(ax=ax2, label="cdf")

    #ax3.set_title(feature_name)
    #ax3.set_xlabel("count")
    #ax3.set_ylabel("PDF")

    #results.plot_pdf(ax=ax1, label="pdf")



def likes(events_yes, events_no):
    feature_events_no = events_no['likes']
    feature_events = events_yes['likes']

    feature_hist(feature_events, feature_events_no, 'likes', cut_from=CUT_FROM)

    #analyze(feature_events_no, feature_name='likes-no events_yes')
    #analyze(feature_events, feature_name='retwelikesets-events_yes')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% group level features %%%%%%%%%%%%%%%%%%%%%%%%
def groupExtractFeatures(group_feature_set, feature_names):
    print("query event groups...")
    q = 'SELECT * FROM group_feature_set WHERE is_event == 1'
    feature_events = sqldf(q, locals())

    print("query none event groups...")
    q = 'SELECT * FROM group_feature_set WHERE is_event == 0'
    feature_events_no = sqldf(q, locals())
    print("# groups: ", len(feature_events) , len(feature_events_no))

    for feature_name in feature_names:
        print("handle feature:", feature_name)
        feature_hist(feature_events[feature_name], feature_events_no[feature_name], feature_name, cut_from=CUT_FROM)

def groupSizeFeature(events_yes, events_no):
    groups_no = events_no.groupby('root')
    groups_yes = events_yes.groupby('root')

    agg_events_no = groups_no.agg(['count'])
    agg_events = groups_yes.agg(['count'])

    feature_events = agg_events['id']['count']
    feature_events_no = agg_events_no['id']['count']
    feature_hist(feature_events, feature_events_no, 'groups1', cut_from=CUT_FROM)


def findBigComponents(tweets, title):
    print('find a top 5 big component:')
    components = sqldf("SELECT root, COUNT(*) AS count FROM tweets GROUP BY root;", locals())
    components = sqldf("SELECT root, count FROM components ORDER BY count DESC;", locals())

    # components['ranked'] = components['count'].rank(ascending=-1)
    print("count per topic:", components.head())
    plotGraphSet(tweets, components, title + ': biggest 4 component')


def findDeepComponents(tweets, title):
    print(title, 'find a top 5 deepest component:')
    components = sqldf("SELECT root, depth FROM tweets WHERE depth in (SELECT MAX(depth) FROM tweets);", locals())

    # components['ranked'] = components['count'].rank(ascending=-1)
    print("count per topic:", components.head())
    plotGraphSet(tweets, components, title + ": deepest 4 component")

def plotGraphSet(tweets, components, title):
    if len(components) == 0:
        print(title, 'no data to plot as graph')
        return

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()
    fig.suptitle(title)

    i = 0
    while i<4 and i<len(components):
        plotGraph(tweets, components['root'][i], ax=axes[(int)(i/2), i%2]) # "'94807275264413697") #""'93983936493010944")
        i+=1

    if SHOW_ONLY:
        fig.show()
    else:
        global img_idx
        savefig(fig)
        img_idx += 1
    plt.close()


def groupFeatures(dataset, events_yes, events_no):
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
                       'MAX(topic_id)>-1 AS is_event, '
                       'SUM(CASE WHEN parentType=2 THEN 1 ELSE 0 END) AS internal_rtwts, '
                       'SUM(CASE WHEN parentType=1 THEN 1 ELSE 0 END) AS internal_rplys '
                       "FROM dataset "
                       "GROUP BY root;", locals())
    print("writing features to csv file...")
    components.to_csv(FOLDER+"/features_" + str(CUT_FROM) + "_" + SUFFEX + "_" + VERSION +".csv", sep="\t")

    if GENERATE_GRAPH:
        findDeepComponents(events_no, "w/o events")
        findBigComponents(events_no, "w/o events")

        findDeepComponents(events_yes, "with events")
        findBigComponents(events_yes, "with events")

    #print('group by root field')
    #groups_no = events_no.groupby('root')
    #groups_yes = events_yes.groupby('root')

    #print('look for max and counts')
    #df_grouped = events_no.groupby('root').agg('count')
    #df_grouped = df_grouped.reset_index()
    #df_grouped = df_grouped.rename(columns={'count': 'count_max'})
    #df = pandas.merge(events_no, df_grouped, how='left', on=['root'])

    #print(df.tail())
    #df = df[df['count'] == df['count_max']]

    #print("Groups: ", df , sep='\t')
    #groupSizeFeature(events_yes, events_no)
    groupExtractFeatures(components, ['size', 'likes_avg', 'depth_max', 'users_count'])

    print("done 'groupFeatures'")

    #fig, ax = plt.subplots(figsize=(8, 6))
    #for label, df in groups:
    #    df.vals.plot(kind="kde", ax=ax, label=label)
    #plt.legend()



def plotGraph(dataset, root, ax):
    q = 'SELECT id, parent, parentType FROM dataset WHERE root == "%s"' % root
    component = sqldf(q, locals())
    #component = dataset[dataset['root'] == root]
    #print("Found : " , len(component), " related items")

    G = nx.DiGraph()
    #component.to_csv('c:/temp/%s.csv' % root, sep=',')
    #print(component)
    G.add_edges_from(component[['id', 'parent']].values, weight=1)
    #G.add_edges_from([('D', 'A'), ('D', 'E'), ('B', 'D'), ('D', 'E')], weight=2)
    #G.add_edges_from([('B', 'C'), ('E', 'F')], weight=3)
    #G.add_edges_from([('C', 'F')], weight=4)

    val_map = {'id': 1.0,
               'parent': 0.5}

    values = [val_map.get(node, 0.45) for node in G.nodes()]

    #edge_labels = parentType
    #edge_labels = dict([((u, v,), d['weight'])
    #                    for u, v, d in G.edges(data=True)])
    #red_edges = [('C', 'D'), ('D', 'A')]
    #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

    pos = nx.spring_layout(G)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #print(ax)
    nx.draw(G, pos=pos, ax=ax, with_labels=True, font_size=8, node_color=values, edge_cmap=plt.cm.Reds)
    labels = G.nodes()
    #labels = {}
    #labels = {k : k for k in labels}
    #nx.draw_networkx_labels(G, pos, labels, font_size=8)
    #plt.axis('off')
    #nx.draw(G, pos, node_color=values, node_size=1500, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
    #pylab.hold(True)
    #pylab.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (end) group level features %%%%%%%%%%%%%%%%%%%%%%%%


def feature_hist(feature_events, feature_events_no, feature_name, cut_from=None):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes.flatten()

    bins = 50

    hist (feature_events_no, ax0, feature_name+' - no events (hist)', feature_name, 'count', bins=bins, color="red", cut_from=cut_from)
    #hist1(feature_events_no, ax2, feature_name+' - no events - hist1', feature_name, 'count', bins=bins, color="red")

    #hist(feature_events_no, ax2, feature_name+' - no events', feature_name, 'count2', bins=bins, color="red")
    #hist(feature_events_no, ax4, feature_name+' (log) - no events', feature_name, 'log(count)', log=True, bins=bins, color="red")

    hist (feature_events, ax1, feature_name+' - events (hist)', feature_name, 'count', bins=bins, color="blue", cut_from=cut_from)
    #hist1(feature_events, ax3, feature_name+' - events- hist1', feature_name, 'count', bins=bins, color="blue")
    #hist(feature_events, ax3, feature_name+' - events', feature_name, 'count2', bins=bins, color="blue")
    #hist(feature_events, ax3, feature_name+' (normalized) - events', normed=1, bins=bins, color="blue")
    #hist(feature_events, ax5, feature_name+' (log) - events', feature_name, 'log(count)', bins=bins, log=True, color="blue")

    # weights = np.ones_like(feature_events_no) / len(feature_events_no)
    # ax2.hist(feature_events_no, bins=bins, weights=weights, histtype='bar', color="red")

    fig.tight_layout()
    if SHOW_ONLY:
        plt.show()
    else:
        global img_idx
        savefig(fig)
        img_idx += 1
    plt.close()

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax0, ax1, ax2, ax3 = axes.flatten()
    #fig, axes = plt.subplots(nrows=3, ncols=2)
    #ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

    analyze(feature_events_no, feature_name + ' events-NO', ax0, ax2)
    analyze(feature_events, feature_name + ' events-YES', ax1, ax3)

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    #handles , labels = ax1.get_legend_handles_labels()
    #ax1.legend( handles , labels )

    #counts, bins = np.histogram(feature_events_no, bins=100)
    #cdf = np.cumsum(counts)
    ##pdf = np.cum.cumsum(counts)
    ##ccdf = np.cumsum(counts)
    ## fig = plt.figure()
    #ax2.plot(bins[1:], cdf)

    fig.tight_layout()

    if SHOW_ONLY:
        plt.show()
    else:
        savefig(fig)
        img_idx += 1
    plt.close()


    print("Plotted features for", feature_name)


def analyzeDataset(filename, sep=','):
    np.seterr(divide='ignore', invalid='ignore')

    print('loading ', filename)
    dataset = pandas.read_csv(filename, sep=sep)
    #ll = locals()
    #pysqldf = lambda q: sqldf(q, ll)
    #print (pysqldf("SELECT * FROM dataset LIMIT 10;").head() )


    #dataset['jRtwt'] = dataset['jRtwt'].astype(str)

    #dataset[dataset['retweets'] == 0] = 0.001

    #print(len(dataset[dataset['retweets'] == 0]))
    #exit()

    #analyze(dataset['likes'], 'likes')
    #exit()
    print('identify topics...')
    #events_no = dataset[dataset['topic_id'] == -1]
    events_no = sqldf("SELECT * FROM dataset WHERE topic_id == -1;", locals())

    #events_yes = dataset[dataset['topic_id'] > -1]
    events_yes = sqldf("SELECT * FROM dataset WHERE topic_id  > -1;", locals())

    print("summary: ")

    #groups = dataset.groupby(by='topic_id')
    #agg = groups.agg(['count', 'max'])
    #print(agg['depth'], sep='\t')

    groups2 = sqldf("SELECT topic_id, COUNT(*) AS count, MAX(depth) AS maxDepth FROM dataset GROUP BY topic_id;", locals())
    print("count per topic:", groups2)

    print("found", len(events_yes), "tweets with", len(groups2)-1, "events")
    print("found", len(events_no), "tweets without any specific events")

    groupFeatures(dataset, events_yes, events_no)

    retweets(events_yes, events_no)
    likes(events_yes, events_no)

if __name__ == "__main__":

    analyzeDataset('c:/temp/dataset_' + SUFFEX + "_" + VERSION + '.txt', sep=',')
    #analyzeDataset('c:/temp/dataset1.2.txt')
    #analyzeDataset('c:/temp/dataset1.4.txt')
