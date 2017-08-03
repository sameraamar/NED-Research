# Authors: Fernando Nogueira
#          Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
from imblearn.over_sampling import SMOTE

def sample(data, id_field_name, y_field_name, sample_size=-1):
    data1 = data[data[y_field_name] == 1 ]
    data2 = data[data[y_field_name] == 0 ]

    if sample_size>0:
        #sample_size = min(sample_size, len(data1))
        #sample_size = min(sample_size, len(data2))
        if sample_size < len(data1) :
            data1 = data1.sample(sample_size) #.head(sample_size)

        if sample_size < len(data2):
            data2 = data2.sample(sample_size) #.head(sample_size)

    xtmp = data1.append(data2)
    cols = [col for col in data.columns if col not in [id_field_name, y_field_name]]

    X = xtmp[cols]
    y = xtmp[y_field_name]

    return X, y

def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

    return c0, c1


# Generate the dataset
#X, y = make_classification(n_classes=2, class_sep=2, weights=[0.3, 0.7],
#                           n_informative=3, n_redundant=1, flip_y=0,
#                           n_features=20, n_clusters_per_class=1,
#                           n_samples=80, random_state=10)

folder = 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/'

print('*' * 20)
print('Tree features')
data = pd.read_csv(folder + 'tree_features.csv', sep=',', index_col=0)

y_field_name = 'is_tree_event'
id_field_name = 'root_id'

X, y = sample(data, id_field_name, y_field_name)

print(X.shape, y.shape)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
print('PCA - init')
# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)
print('PCA - fit')

# Apply regular SMOTE
kind = ['regular', 'borderline1', 'borderline2', 'svm']
sm = [SMOTE(kind=k) for k in kind]
X_resampled = []
y_resampled = []
X_res_vis = []
for method in sm:
    print('Method: ', method)
    X_res, y_res = method.fit_sample(X, y)
    X_resampled.append(X_res)
    y_resampled.append(y_res)
    X_res_vis.append(pca.transform(X_res))

# Two subplots, unpack the axes array immediately
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
# Remove axis for second plot
ax2.axis('off')
ax_res = [ax3, ax4, ax5, ax6]

c0, c1 = plot_resampling(ax1, X_vis, y, 'Original set')
for i in range(len(kind)):
    plot_resampling(ax_res[i], X_res_vis[i], y_resampled[i],
                    'SMOTE {}'.format(kind[i]))

ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center',
           ncol=1, labelspacing=0.)
plt.tight_layout()
plt.show()