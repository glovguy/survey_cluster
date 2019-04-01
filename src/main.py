import json
from models.surveys import Surveys
from models.clusters import KmeansClusters
from cluster_diagram import generate_diagram
from silouette_scores import find_best_clust_num


data_raw = [
    {'survey_text': 'The quick brown fox jumps over the lazy dog', 'surveyid': '1'},
    {'survey_text': 'Some days require more coffee than others', 'surveyid': '2'},
    {'survey_text': 'coffee makes the world go round', 'surveyid': '3'},
    {'survey_text': 'Crazy like a fox', 'surveyid': '4'}
]
infer_num_clusters = False
num_clusters = 2

srvys = Surveys(data_raw)
kmObject = KmeansClusters(srvys)
kmObject.set_num_clusters(num_clusters) if infer_num_clusters else find_best_clust_num(kmObject, save_plot=True)
generate_diagram(srvys, kmObject)
print(json.dumps(kmObject.clusters_to_dict()))
