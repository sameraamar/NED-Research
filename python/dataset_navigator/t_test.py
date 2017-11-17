from scipy.stats import ttest_ind, ttest_rel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from statsmodels.stats.power import  tt_ind_solve_power, ttest_power
from scipy.interpolate import interp1d

'''If you have two independent samples but you do not know that they have equal variance, you can use Welch's t-test. It is as simple as

     scipy.stats.ttest_ind(cat1['values'], cat2['values'], equal_var=False)
For reasons to prefer Welch's test, see https://stats.stackexchange.com/questions/305/when-conducting-a-t-test-why-would-one-prefer-to-assume-or-test-for-equal-vari.

We can use this test, if we observe two independent samples from the same or different population, e.g. exam scores of boys and girls or of two ethnic groups.
The test measures whether the average (expected) value differs significantly across samples.
If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores.
If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.

For two dependent samples, you can use:
    scipy.stats.ttest_rel(cat1['values'], cat2['values'])
Examples for the use are scores of the same set of student in different exams, or repeated sampling from the same units.
The test measures whether the average score differs significantly across samples (e.g. exams).
If we observe a large p-value, for example greater than 0.05 or 0.1 then we cannot reject the null hypothesis of identical average scores.
If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
Small p-values are associated with large t-statistics.

'''

CTR_csv = 'C:/Users/t-saaama/OneDrive - Microsoft/Work/Tasks/Expirement/KPI_CTR_flat10-01.40_days.csv'

from scipy.stats import norm, zscore
def sample_power_probtest(p1, p2, power=0.8, sig=0.05):
    z = norm.isf([sig/2]) #two-sided t test
    zp = -1 * norm.isf([power])
    d = (p1-p2)
    s =2*((p1+p2) /2)*(1-((p1+p2) /2))
    n = s * ((zp + z)**2) / (d**2)
    return int(round(n[0]))

def sample_power_difftest(d, s, power=0.8, sig=0.05):
    z = norm.isf([sig/2])
    zp = -1 * norm.isf([power])
    n = s * ((zp + z)**2) / (d**2)
    return int(round(n[0]))


#def sample_power_difftest(d, s, power=0.8, sig=0.05):
#    z = norm.isf([sig/2])
#    zp = -1 * norm.isf([power])
#    n = (2*(s**2)) * ((zp + z)**2) / (d**2)
#    return int(round(n[0]))

def descriptive_statistics(title, data):
    print(title + ': ', data.head(10))
    print(data.describe())


def test_ttest_power_diff(mean, std, sample1_size=None, alpha=0.05, desired_power=0.8, mean_diff_percentages=[0.1, 0.05]):
    '''
    calculates the power function for a given mean and std. the function plots a graph showing the comparison between desired mean differences
    :param mean: the desired mean
    :param std: the std value
    :param sample1_size: if None, it is assumed that both samples (first and second) will have same size. The function then will
    walk through possible sample sizes (up to 100, hardcoded).
    If this value is not None, the function will check different alternatives for sample 2 sizes up to sample 1 size.
    :param alpha: alpha default value is 0.05
    :param desired_power: will use this value in order to mark on the graph
    :param mean_diff_percentages: iterable list of percentages. A line per value will be calculated and plotted.
    :return: None
    '''
    fig, ax = plt.subplots()
    for mean_diff_percent in mean_diff_percentages:
        mean_diff = mean_diff_percent * mean
        effect_size = mean_diff / std

        print('Mean diff: ', mean_diff)
        print('Effect size: ', effect_size)

        powers = []

        max_size  = sample1_size
        if sample1_size is None:
            max_size = 100

        sizes = np.arange(1, max_size, 2)
        for sample2_size in sizes:
            if(sample1_size is None):
                n = tt_ind_solve_power(effect_size=effect_size, nobs1=sample2_size, alpha=alpha, ratio=1.0, alternative='two-sided')
                print('tt_ind_solve_power(alpha=', alpha, 'sample2_size=', sample2_size, '): sample size in *second* group: {:.5f}'.format(n))
            else:
                n = tt_ind_solve_power(effect_size=effect_size, nobs1=sample1_size, alpha=alpha, ratio=(1.0*sample2_size/sample1_size), alternative='two-sided')
                print('tt_ind_solve_power(alpha=', alpha, 'sample2_size=', sample2_size, '): sample size *each* group: {:.5f}'.format(n))

            powers.append(n)

        try:
            z1 = interp1d(powers, sizes)
            results = z1(desired_power)

            plt.plot([results], [desired_power], 'gD')
        except Exception as e:
            print("Error: ", e)
            #ignore

        plt.title('Power vs. Sample Size')
        plt.xlabel('Sample Size')
        plt.ylabel('Power')

        plt.plot(sizes, powers, label='diff={:2.0f}%'.format(100*mean_diff_percent)) #, '-gD')

    plt.legend()
    plt.show()



# def test_ttest_power(mean_diff, sd_diff, sd_avg):
#     std_effect_size = mean_diff / sd_diff
#
#     print('Mean diff: ', mean_diff)
#     print('STD diff: ', sd_diff)
#     print('Effect size: ', std_effect_size)
#
#     fig, ax = plt.subplots()
#     for alpha in [0.05, 0.1]:
#         powers = []
#         sizes = np.arange(5, 100, 5)
#         for sampleSize in sizes:
#             # effect_size=None, nobs1=None, alpha=None, power=None, ratio=1., alternative='two-sided'
#             n = tt_ind_solve_power(effect_size=mean_diff / sd_avg, nobs1=sampleSize, alpha=alpha, ratio=1, alternative='two-sided')
#             print('tt_ind_solve_power(alpha=', alpha, 'sample_size=', sampleSize, '): Number in *each* group: {:.5f}'.format(n))
#             powers.append(n)
#
#         try:
#             z1 = interp1d(powers, sizes)
#             results = z1(0.8)
#
#             #plt.plot([results, results], [np.min (powers) , 0.8], '--', c='r')
#             #plt.plot([0, results], [0.8, 0.8], '--', c='r')
#             plt.plot([results], [0.8], 'gD')
#
#             #ax.spines['left'].set_position('zero')
#             #ax.spines['bottom'].set_position('zero')
#         except Exception as e:
#             print("Error: ", e)
#             #do nothing
#
#         plt.title('Power vs. Sample Size')
#         plt.xlabel('Sample Size')
#         plt.ylabel('Power')
#
#         plt.plot(sizes, powers, label='alpha={:2.0f}%'.format(100*alpha)) #, '-gD')
#
#     plt.legend()
#     plt.show()


def CTR():
    data = pd.read_csv(CTR_csv)
    my_data = data.sort_values(['DateId'], ascending=[True])

    #my_data.groupby('Expirement').mean()
    #cat1 = my_data['ExpA']
    #cat2 = my_data['ExpB']

    ##cat1 = cat1.sort_values(by=['DateId'])
    ##cat2 = cat2.sort_values(by=['DateId'])

    dates = my_data['DateId']
    expirementA = my_data['ExpA']
    expirementB = my_data['ExpB']

    #mu, sigma = 0.065939, 0.002768  # mean and standard deviation
    #expirementA = pd.DataFrame(np.random.normal(mu, sigma, 31), columns=['ExpA'])['ExpA']
    #mu, sigma = 0.069284, 0.003688  # mean and standard deviation
    #expirementB = pd.DataFrame(np.random.normal(mu, sigma, 31), columns=['ExpB'])['ExpB']

    descriptive_statistics('Expirement A: ', expirementA)
    descriptive_statistics('Expirement B: ', expirementB)

    plot_data(dates, expirementA, expirementB)
    plot_box_whisker(dates, expirementA, expirementB)
    calc_power_function(dates, expirementA, expirementB)

def calc_power_function(dates, expirementA, expirementB):
    std_avg = np.std(expirementA) + np.std(expirementB)
    std_avg /= 2.0

    #test_ttest_power( np.abs( np.mean(expirementA)-np.mean(expirementB) ), np.abs( np.std(expirementA)-np.std(expirementB) ), std_avg )
    test_ttest_power_diff( np.mean(expirementA), np.std(expirementA), len(expirementA)+1, mean_diff_percentages=[0.15, 0.1, 0.05] )
    test_ttest_power_diff( np.mean(expirementA), np.std(expirementA), mean_diff_percentages=[0.15, 0.1, 0.05] )

    print(ttest_ind(expirementA, expirementB, equal_var=False))
    print(ttest_rel(expirementA, expirementB))

    expirementA.hist(histtype='barstacked', label="ExpA", alpha=0.5)
    ax = expirementB.hist(histtype='stepfilled', label="ExpB",  alpha=0.5)

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:3.1f}%'.format(x*100) for x in vals])

    plt.title('CTR histogram')
    plt.xlabel('CTR')
    plt.ylabel('count')

    plt.legend()
    plt.show()


def plot_box_whisker(dates, expirementA, expirementB):
    #ax = pd.DataFrame().plot()
    #vals = ax.get_yticks()
    #ax.set_yticklabels(['{:3.1f}%'.format(x*100) for x in vals])

    ax = pd.concat([expirementA, expirementB], axis=1).boxplot()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.1f}%'.format(x*100) for x in vals])

    #expirementA.plot(kind='box', subplots=True, label="A", sharex=False, sharey=False)
    #expirementB.plot(kind='box', subplots=True, label="B", sharex=False, sharey=False)

    plt.title('CTR mean/std')
    plt.xlabel('Experiment')
    plt.ylabel('CTR')

    #plt.legend()
    plt.show()

def plot_data(dates, expirementA, expirementB):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    idates = np.arange(1, len(dates)+1, 1)
    ax.plot(idates, expirementA, '.r-')
    ax.plot(idates, expirementB, 'xb-')

    ytext = np.min( [np.min( expirementA ), np.min( expirementB )] )

    for i in idates:
        if i%3 == 0:
            ax.text(i, ytext, dates[i-1], fontsize='x-small', rotation=60, va='bottom' )

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.1f}%'.format(x*100) for x in vals])
    vals = ax.get_xticks()
    ax.set_xticklabels(['{0}'.format(x) for x in vals])

    plt.title('CTR per day')
    plt.xlabel('Day')
    plt.ylabel('CTR')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_ttest_power_diff(10, 2, mean_diff_percentages=[0.05, 0.1, 0.15, 0.2] )
    test_ttest_power_diff(10, 2, mean_diff_percentages=[0.05, 0.1, 0.15] )
    test_ttest_power_diff(10, 2, mean_diff_percentages=[0.05, 0.1] )
    test_ttest_power_diff(10, 2, mean_diff_percentages=[0.05] )
    CTR()
