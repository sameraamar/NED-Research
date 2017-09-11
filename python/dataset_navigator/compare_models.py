import subprocess

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
        error = error.decode(encoding='UTF-8', errors='replace')

        #print (error)

    return result, error
# def

def read_gpd_info(gpdpath, catpath, seedspath, outpath):
    command =  ["del", outpath]
    #print('... ' , *execute(command))

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

from compare_lists import plot_res, compare_files

if __name__ == "__main__":
    flag = False
    left_model_file = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\v2v_long.txt'
    right_model_file = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\p2p_long.txt'

    if flag:
        seedspath = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\seeds_apps_small.txt'
        if not prepareV2V(seedspath, left_model_file):
            exit()

        if not prepareP2P(seedspath, right_model_file):
            exit()

    res = compare_files(left_model_file, right_model_file)
    plot_res(res)


























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

