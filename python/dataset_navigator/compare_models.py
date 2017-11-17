import subprocess
import pandas as pd
from compare_lists import plot_res, compare_files

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

    return rc == 0, result, error
# def

def read_gpd_info(gpdpath, binpath, catpath, seedspath, outpath):
    command =  ["del", outpath]
    #print('... ' , *execute(command))

    tmp = gpdpath.split('\\')
    tmp = tmp[len(tmp)-1].split('/')
    tmp = tmp[len(tmp)-1].split('.')
    version = tmp[0]

    command = [ITEM_GPD_READER_EXE, '-v', version, '-gpd', gpdpath, '-bin', binpath, '-cat', catpath, "-count", "35", "-o", outpath, seedspath] #, '>', quote(outpath)]
    success, result, error = execute(command)

    print('res:', result)
    print('err:', error)

    #' -bin C:\data\Work\Source\Repos\SC.RecoModeling\Drop\Release\AnyCPU\Source\Offline\ExplorationTool\PgProduct\itemModel_Reco_20170719082438.bin ' \
    return success

def check_coverage(ids_df, recommended_file):
    recommended = pd.read_csv(recommended_file, sep=",", header=0)
    missing = ids_df[~ids_df['ItemId'].isin(recommended['Seed(Guid)'])]['ItemId'].to_frame()['ItemId'].tolist()

    print('Items not covered in (',recommended_file ,'): ', len(missing))
    print(missing)

    #temp = ids_df['ItemId'].isin(missing)
    #ids_df.c1[a.c1 == 8].index.tolist()

def Apps_prepareV2V(seedspath, outpath):
    path    = 'C:\\temp\\models\\DesktopApps-V2V modeling'

    gpdpath = path + '\\Reco_20170822100829.GlobalPredictionData'
    catpath = path + '\\catalog.csv'
    binpath = path + '\\itemModel.bin'

    return read_gpd_info(gpdpath, binpath, catpath, seedspath, outpath)

def Apps_prepareP2P(seedspath, outpath):
    path    = 'C:\\temp\\models\\DesktopApps-P2P modeling'

    gpdpath = path + '\\Reco_20170822093207.GlobalPredictionData'
    catpath = path + '\\catalog.csv'
    binpath = path + '\\itemModel.bin'

    return read_gpd_info(gpdpath, binpath, catpath, seedspath, outpath)

def Apps_read_seeds1(outputseedpath=None):
    seedspath = 'C:\\Users\\t-saaama\\OneDrive - Microsoft\\Work\\Tasks\\General\\MostPopularSeedsItems_DeskApps.ss.csv'
    if outputseedpath is None:
        outputseedpath = seedspath + '.guid'

    seeds = pd.read_csv(seedspath, sep=",", header=0)
    seeds = seeds.sort_values(by='Popularity', ascending=False)
    print(seeds.ix[:, ['ItemId', 'ProductTitle']].head(10))
    guids = seeds.head(1000)['ItemId']
    guids.to_csv(outputseedpath, sep=",", index=False, header=None)

    return outputseedpath, guids.to_frame()


def Games_prepareV2V(seedspath, outpath):
    path    = 'C:\\temp\\models\\DesktopGames-V2V\\Games'

    gpdpath = path + '\\Reco_20170920130424.GlobalPredictionData'
    catpath = path + '\\WinDesktopGamesUserClicksModeling_catalogEffective.csv'
    binpath = path + '\\WinDesktopGamesUserClicksModeling_itemModel.bin'

    return read_gpd_info(gpdpath, binpath, catpath, seedspath, outpath)

def Games_prepareP2P(seedspath, outpath):
    path    = 'C:\\temp\\models\\DesktopGames-P2P'

    gpdpath = path + '\\Reco_20170927093239.GlobalPredictionData'
    catpath = path + '\\catalog.csv'
    binpath = path + '\\itemModel.bin'

    return read_gpd_info(gpdpath, binpath, catpath, seedspath, outpath)


def Games_read_seeds2(outputseedpath=None):
    seedspath = 'C:\\temp\\models\\games_catalog_mostpopular_with_popularity.csv'
    if outputseedpath is None:
        outputseedpath = seedspath + '.guid'

    seeds = pd.read_csv(seedspath, sep=",", header=0)
    seeds = seeds.sort_values(by='Popularity', ascending=False)
    print(seeds.ix[:, ['ItemId', 'ProductTitle']].head(10))
    guids = seeds.head(1000)['ItemId']
    guids.to_csv(outputseedpath, sep=",", index=False, header=None)

    return outputseedpath, guids.to_frame()


def Games_read_seeds1(outputseedpath=None):
    seedspath = 'C:\\temp\\models\\Games_mostpopular.ss.csv'
    if outputseedpath is None:
        outputseedpath = seedspath + '.guid'

    if seedspath == outputseedpath:
        print("Error: output path is same as input: ", seedspath)

    seeds = pd.read_csv(seedspath, sep=",", header=0)
    guids = seeds.head(1000)['ItemId']
    print('working on ', len(guids), 'items: ', seeds.ix[:, ['ItemId', 'ProductTitle']].head(10))
    guids.to_csv(outputseedpath, sep=",", index=False, header=None)

    return outputseedpath, guids.to_frame()

if __name__ == "__main__":
    flag = False
    mode = 0 # 0 - Apps, 1 - Games, 2 - Physical Goods (alpha 0.25), 3 - Phyusical Goods (alpha 0.35)
    titles = ['left', 'right']

    if mode==0:
        titles = ['V2V', 'P2P']

        left_model_file = 'C:\\temp\\v2v_long_popular.csv'
        right_model_file = 'C:\\temp\\p2p_long_popular.csv'

    elif mode==1:
        titles = ['V2V', 'P2P']
        left_model_file = 'C:\\temp\\games_v2v_long_popular3.csv'
        right_model_file = 'C:\\temp\\games_p2p_long_popular3.csv'

    elif mode == 2:
        titles = ['LRC', 'DAS(0.25)']
        left_model_file = 'C:\\temp\\pg\\lrc_pg_clean.csv'
        right_model_file = 'C:\\temp\\pg\\das_alpha0.25_clean.csv'

    else: # mode == 3
        titles = ['LRC', 'DAS(0.35)']
        left_model_file = 'C:\\temp\\pg\\lrc_pg_clean.csv'
        right_model_file = 'C:\\temp\\pg\\das_alpha0.35_clean.csv'


    #/////////
    # Example:
    # seedspath = 'C:\\temp\\DesktopApps_V2V_vs_P2P\\seeds_apps_small.txt'

    # read the top item list

    seedspath = ''
    guids = []
    if mode==0:
        seedspath, guids = Apps_read_seeds1('C:/temp/MostPopularSeedsItems_DeskApps.guid.txt')

        if flag:
            if not Apps_prepareV2V(seedspath, left_model_file):
                exit()

            if not Apps_prepareP2P(seedspath, right_model_file):
                exit()


    elif mode == 1:
        seedspath, guids = Games_read_seeds2('c:/temp/games_guid.guid_short.guid.txt')

        if flag:
            if not Games_prepareV2V(seedspath, left_model_file):
                exit()

            if not Games_prepareP2P(seedspath, right_model_file):
                exit()

    else:
        seedspath = 'C:\\temp\\pg\\kernel.csv'
        guids = pd.read_csv(seedspath)

    #/////////

    check_coverage(guids, left_model_file)
    check_coverage(guids, right_model_file)

    res = compare_files(left_model_file, right_model_file, 10, 30)
    plot_res(res, 10, 30, titles)


























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

