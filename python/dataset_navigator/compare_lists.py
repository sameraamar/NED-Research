from matplotlib_venn import venn2
import numpy as np
import pandas as p
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as pr
from scipy import stats
import subprocess



def myfunc(model1, model2, seed):
    a = set(model1.loc[model1['seed_id'] == seed].ix[:, 2].head(12))
    b = set(model2.loc[model2['seed_id'] == seed].ix[:, 2].head(12))

    return len(a), len(b), len(a.intersection(b))

ITEM_GPD_READER_EXE = 'C:\\data\\Work\\Source\\Repos\\SC.RecoModeling\\Source\\Offline\\Item2ItemReader\\bin\\Release\\Item2ItemReader.exe'

def quote(filename):
    return '"' + filename + '"'

def execute(cmd, out=None):
    """
        Purpose  : To execute a command and return exit status
        Argument : cmd - command to execute
        Return   : exit_code
    """

    if out is None:
        output = subprocess.PIPE
    else:
        output = open(out, "w", encoding="utf-8")

    process = subprocess.Popen(cmd, shell=True, stdout=output, stderr=subprocess.PIPE)
    print('echo: ', ' '.join(cmd))

    (result, error) = process.communicate()

    rc = process.wait()

    #if (error is None):
    #    error = ''

    #if (result is None):
    #    result = ''

    #error = error.decode(encoding='UTF-8',errors='strict').strip()
    #result = result.decode(encoding='UTF-8',errors='strict')

    if rc != 0 or (error is not None and error != ""):
        #print ("Error: failed to execute command:")
        error = error.decode(encoding='UTF-8',errors='strict')
        #print (error)

    return result, error
# def

def read_gpd_info(gpdpath, catpath, seedspath, outpath):
    command =  ["del", outpath]
    print('... ' , *execute(command))


    tmp = gpdpath.split('\\')
    tmp = tmp[len(tmp)-1].split('/')
    tmp = tmp[len(tmp)-1].split('.')
    version = tmp[0]

    command = [ITEM_GPD_READER_EXE, '-v', version, '-gpd', gpdpath, '-cat', catpath, seedspath, outpath] #, '>', quote(outpath)]
    result, error = execute(command)

    print('res:', result)
    print('err:', error)

    #' -bin C:\data\Work\Source\Repos\SC.RecoModeling\Drop\Release\AnyCPU\Source\Offline\ExplorationTool\PgProduct\itemModel_Reco_20170719082438.bin ' \
    return True

def prepareV2V(seedspath, outpath):
    path    = 'C:\\temp\\DesktopApps-V2V modeling'

    gpdpath = path + '\\Reco_20170822100829.GlobalPredictionData'
    catpath = path + '\\catalog.csv'

    return read_gpd_info(gpdpath, catpath, seedspath, outpath)

def prepareP2P(seedspath, outpath):
    path    = 'C:\\temp\\DesktopApps-P2P modeling'

    gpdpath = path + '\\Reco_20170822093207.GlobalPredictionData'
    catpath = path + '\\catalog.csv'

    return read_gpd_info(gpdpath, catpath, seedspath, outpath)

if __name__ == "__main__":
    flag = True
    v2vpath = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\v2v.txt'
    p2ppath = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\p2p.txt'

    if flag:
        seedspath = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\seeds_apps.txt'
        if not prepareV2V(seedspath, v2vpath):
            exit()

        if not prepareP2P(seedspath, p2ppath):
            exit()

    names = ['seed_id', 'seed_title', 'reco_id', 'reco_title']
    v2v = p.read_csv(v2vpath, sep=",", header=None, names=names)
    v2v_pairs = [x for x in zip(v2v.ix[:, 0], v2v.ix[:, 2])]


    p2p = p.read_csv(p2ppath, sep=",", header=None, names=names)
    p2p_pairs = [x for x in zip(p2p.ix[:, 0], p2p.ix[:, 2])]

    print(v2v.shape, v2v.head(5))
    print(p2p.shape, p2p.head(5))


    set_v2v = set(v2v_pairs)
    set_p2p = set(p2p_pairs)

    print('V2V', len(set_v2v),
          ', P2P', len(set_p2p),
          ', intersection: ', len( set_v2v.intersection( set_p2p ) ),
          ', V2V-P2P:', len( set_v2v.difference( set_p2p )),
          ', P2P-V2V:', len( set_p2p.difference( set_v2v )))


    a = np.array(set_p2p)
    b = np.array(set_v2v)

    error = np.mean( a != b )
    error = (a != b).sum()/float(a.size)
    print(error)

    venn2([set(v2v_pairs), set(p2p_pairs)], set_labels=('V2V', 'P2P'))
    plt.show()


    seeds = set( [x for (x,y) in v2v_pairs] )


    aa = [(seed, *myfunc(v2v, p2p, seed)) for seed in seeds]
    print(aa)

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

