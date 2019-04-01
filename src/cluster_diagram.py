import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity


def generate_diagram(srvys, kmClusters):
    dist = 1 - cosine_similarity(srvys.tfidf_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)

    xs, ys = pos[:, 0], pos[:, 1]
    cluster_names = {i: kmClusters.feature_phrases_for_cluster(i, max_n=3) for i in range(0, kmClusters.num_clusters)}
    df = pd.DataFrame(dict(x=xs, y=ys, label=kmClusters.cluster_labels()))
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y,
                marker='o',
                linestyle='',
                ms=12,
                label=cluster_names[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        ax.tick_params(\
            axis= 'y',
            which='both',
            left=False,
            top=False,
            labelleft=False)
    ax.legend(numpoints=1)
    # plt.show()
    plt.savefig('clusters_small_noaxes.png', dpi=200)

