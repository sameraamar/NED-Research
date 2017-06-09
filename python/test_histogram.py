import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import numpy as np
import powerlaw
import pandas as pd

def testHist():
    mu, sigma = 0, 100
    x = mu + sigma * np.random.rand(100)


    #x= np.log10(x)
    bins = 30 #[0, 40, 60, 75, 90, 110, 125, 140, 160, 200]
    hist, bins = np.histogram(x, bins=bins)
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    #fig.savefig("/tmp/out.png")

    plt.show()


def testSurface(dataset):
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


dataset = pd.read_csv('c:/temp/performance.csv', sep=',')
print(dataset.info())
testSurface(dataset)


