import matplotlib.pyplot as plt


def find_best_clust_num(kmObject, save_plot=False):
    sil_dict = kmObject.best_num_clusters()
    best_num = max(sil_dict, key=sil_dict.get)
    if save_plot: plot_sillouette_scores(sil_dict)
    kmObject.set_num_clusters(best_num)

def plot_sillouette_scores(sil_dict):
    x = sil_dict.keys()
    y = [sil_dict[i] for i in x]
    plt.plot(x,y)
    plt.savefig('silhouettes.png', dpi=200)
