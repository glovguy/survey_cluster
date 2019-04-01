from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

class KmeansClusters(object):
    def __init__(self, surveys):
        self.surveys = surveys

    def set_num_clusters(self, num_clusters):
        self.num_clusters = num_clusters
        self.km = KMeans(n_clusters=num_clusters)
        self.km.fit(self.surveys.tfidf_matrix)

    def best_num_clusters(self):
        sil_scores = {}
        max_num_clusters = min(30, self.surveys.survey_count()-1)
        for num in range(2, max_num_clusters+1):
            self.set_num_clusters(num)
            sil_scores[num] = self.silhouette()
        print('sil_scores', self.surveys.survey_count(), sil_scores)
        return sil_scores

    def silhouette(self):
        return silhouette_score(self.surveys.tfidf_matrix, self.cluster_labels())

    def cluster_labels(self):
        return self.km.labels_.tolist()

    def feature_names(self):
        return self.surveys.vectorizer.get_feature_names()

    def centroids(self):
        return self.km.cluster_centers_.argsort()[:, ::-1]

    def feature_phrases_for_cluster(self, cluster_i, max_n=0):
        output = []
        centroid = self.centroids()[cluster_i]
        for feature in centroid:
            max_n -= 1
            output.append(self.surveys.feature_names_unstemmed()[feature])
            if max_n == 0: return output
        return output

    def members_of_cluster(self, cluster_i):
        survey_count = len(self.surveys.data_raw)
        return (self.surveys.data_raw[i] for i in range(0, survey_count) if cluster_i == self.cluster_labels()[i])

    def cluster_to_dict(self, cluster_i):
        return {
                'id': cluster_i,
                'count': self.cluster_labels().count(cluster_i),
                'phrases': self.feature_phrases_for_cluster(cluster_i),
                "examples": [m['survey_text'] for m in itertools.islice(self.members_of_cluster(cluster_i), 10)],
                'members': [m['surveyid'] for m in self.members_of_cluster(cluster_i)],
            }

    def clusters_to_dict(self):
        cluster_info = [self.cluster_to_dict(cluster_i) for cluster_i in range(0, self.num_clusters)]
        return {
                "feature_count": len(self.surveys.vectorizer.get_feature_names()),
                "stop_words_count": len(list(self.surveys.vectorizer.get_stop_words())),
                "total_analyzed": self.surveys.survey_count(),
                "cluster_count": self.num_clusters,
                "clusters": sorted(cluster_info, key=lambda c: c['count'], reverse=True),
                "all_features": self.surveys.feature_names_unstemmed(),
                "all_stop_words": list(self.surveys.vectorizer.get_stop_words()),
            }
