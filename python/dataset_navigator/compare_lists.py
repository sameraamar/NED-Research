from matplotlib_venn import venn2
import numpy as np
import pandas as p
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as pr
from scipy import stats
import subprocess
from sklearn.metrics import jaccard_similarity_score

def overlap_check(model1, model2, seed, left_head, right_head):
    a = set(model1.loc[model1['seed_id'] == seed].ix[:, 2].head(left_head))
    b = set(model2.loc[model2['seed_id'] == seed].ix[:, 2].head(right_head))

    return 1.0 * len(a.intersection(b)) / left_head

def jaccard_score(model1, model2, seed, head):
    a = set(model1.loc[model1['seed_id'] == seed].ix[:, 2].head(head))
    b = set(model2.loc[model2['seed_id'] == seed].ix[:, 2].head(head))

    #a = list(a)
    #b = list(b)
    #return jaccard_similarity_score(a, b, normalize=True)

    intersection = len(a.intersection(b))
    union = len(a) + len(b) - intersection

    return intersection / union

def compare(left, right, limit1, limit2):
    seeds = set( [x for x in left['seed_id']] )

    res = [(seed, overlap_check(left, right, seed, limit1, limit1), overlap_check(left, right, seed, limit1, limit2), overlap_check(right, left, seed, limit1, limit2), jaccard_score(right, left, seed, limit1)) for seed in seeds]
    return p.DataFrame(res)

    #return aa, [x for x in zip(*aa)]

def compare_files(left_model_file, right_model_file, limit1=10, limit2=30):
    print('Comparing...')

    names = ['seed_id', 'seed_title', 'reco_id', 'reco_title', 'score']
    left_model = p.read_csv(left_model_file, sep=",", header=None, names=names)
    left_pairs = [x for x in zip(left_model.ix[:, 0], left_model.ix[:, 2], left_model.ix[:, 4])]

    right_model = p.read_csv(right_model_file, sep=",", header=None, names=names)
    right_pairs = [x for x in zip(right_model.ix[:, 0], right_model.ix[:, 2], right_model.ix[:, 4])]

    #print(left_model.shape, left_model.head(5))
    #print(right_model.shape, right_model.head(5))

    res = compare(left_model, right_model, limit1, limit2)
    return res

def plot_res(res, limit1, limit2, model_titles=['left', 'right']):
    #labels = ["Left top 10 <==> Right top 10", "Left top 10 <==> Right top 30", "Right top 10 <==> Left top 30"]

    labels = [model_titles[0]+" top " + str(limit1) + " <==> "+model_titles[1]+" top " + str(limit1),
              model_titles[0] + " top " + str(limit1) + " <==> "+model_titles[1]+" top " + str(limit2),
              model_titles[1]+" top " + str(limit1) + " <==> "+model_titles[0]+" top " + str(limit2),
              "jaccard"]

    colors = ["blue", "red", "green", "magenta"]

    binning = [True, True, True, False]

    bins = 0
    together = []
    for i in range(4):
        i = 3-i

        fig = plt.figure()
        st = fig.suptitle("Recommendation Models Comparison", fontsize="x-large")

        s = res[i+1]

        print(i, ": ", len(s))
        together.append(s)

        maxy = len(s)
        print('maxy = ', maxy)

        bins = None
        if binning[i]:
            bins = int(maxy / 100)
            if maxy < 100:
                bins = int(maxy / 10.0)

        print('bins --> ', bins)

        #ax = fig.add_subplot(311+i)
        ax = fig.add_subplot(111)

        ax.hist(np.asanyarray(s), color=colors[i], bins=bins)
        ax.set_xlim([0, 1])

        if i != 3: #jaccard score
            # manipulate
            vals = ax.get_xticks()
            ax.set_xticklabels(['{:3.0f}'.format(x*10) for x in vals])
            ax.set_xlabel("Number of items in "+model_titles[0]+" that appear in "+model_titles[1])
        else:
            ax.set_xlabel("jaccard score")

        # manipulate
        ax.set_ylabel("items count")
        ax.set_xlabel("Number of items in "+model_titles[0]+" that appear in "+model_titles[1])

        if maxy > 10:
            ax.set_ylabel("items count (%)")
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.0f}%'.format(100.0 * y / maxy) for y in vals])

        ax.set_title(labels[i], fontsize="small")

        print()
        print(labels[i], s.describe())


        fig.tight_layout()

        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
        plt.show()

    plt.hist(together, bins=bins, label=labels, histtype='bar')
    plt.figtext(1.0, 0.2, together[0].describe())

    plt.xlabel('overlap %')
    plt.ylabel('count')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    flag = True
    left_model_file = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\v2v.txt'
    right_model_file = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\p2p.txt'

    res = compare_files(left_model_file, right_model_file)

    print(res.head(10))

    plot_res(res)

    #fig = plt.gcf()
    #plt.show()


    print("-----------------------------------------")

    left_set = set(left_pairs)
    right_set = set(right_pairs)

    print('left_model', len(left_set),
          ', right_model', len(right_set),
          ', intersection: ', len( left_set.intersection( right_set ) ),
          ', left_model-P2P:', len( left_set.difference( right_set )),
          ', right_model-left_model:', len( right_set.difference( left_set )))


    a = np.array(right_set)
    b = np.array(left_set)

    error = np.mean( a != b )
    error = (a != b).sum()/float(a.size)
    print(error)

    venn2([set(left_pairs), set(right_pairs)], set_labels=('left_model', 'right_model'))
    plt.show()


    #intersections1 = [(id,x, int(x/b*100)*1.0) for (id, a, b, x) in aa]
    #intersections = [int(x/b*100)*1.0 for (id, a, b, x) in aa]

    #print(intersections1, intersections)
    #print(stats.describe(intersections))


    #----------------------























def test():
    lrc = p.read_csv('c:/temp/LRC.csv')
    print(len(lrc), lrc.shape)

    #lrc_pairs = lrc.ix[:, 0] + ',' + lrc.ix[:, 2]
    lrc_pairs = [x for x in zip(lrc.ix[:, 0] , lrc.ix[:, 2])]
    #print(lrc_pairs.shape, lrc_pairs.head(10))

    #----------------------

    mf = p.read_csv('c:/temp/reco_items.csv')
    print(len(lrc), lrc.shape)
    print(len(mf), mf.shape)

    #mf_pairs = mf.ix[:, 0] + ',' + mf.ix[:, 2]
    mf_pairs = [x for x in zip(mf.ix[:, 0] , mf.ix[:, 2])]
    #print('mf_pairs' , mf_pairs.head(10))

    #print('LRC (with duplicates)', len(lrc_pairs), lrc_pairs.shape)
    #print('MF  (with duplicates)', len(mf_pairs), mf_pairs.shape)
    #-----------------------

    set_mf = set(mf_pairs)
    set_lrc = set(lrc_pairs)

    print('MF', len(set_mf),
          ', LRC', len(set_lrc),
          ', intersection: ', len( set_mf.intersection( set_lrc ) ),
          ', MF-LRC:', len( set_mf.difference( set_lrc )),
          ', LRC-MF:', len( set_lrc.difference( set_mf )))


    a = np.array(set_lrc)
    b = np.array(set_mf)

    error = np.mean( a != b )
    error = (a != b).sum()/float(a.size)
    print(error)

    venn2([set(mf_pairs), set(lrc_pairs)], set_labels=('MF', 'LRC'))
    plt.show()


    seeds = set( [x for (x,y) in mf_pairs] )


    aa = [(seed, *myfunc(mf, lrc, seed)) for seed in seeds]
    #print(len(aa), aa)

    intersections1 = [(id,x, int(x/b*100)*1.0) for (id, a, b, x) in aa]
    intersections = [int(x/b*100)*1.0 for (id, a, b, x) in aa]

    print(intersections1, intersections)
    print(stats.describe(intersections))

    plt.hist(intersections)
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    fig = plt.gcf()
    plt.show()


    #bPrecis, bRecall, bFscore, bSupport = pr(mf_pairs, lrc_pairs, average='binary')
    #print(bPrecis, bRecall, bFscore, bSupport)

