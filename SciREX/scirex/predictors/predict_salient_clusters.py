import json
import sys

def predict(clusters_file, saliency_file, output_file):
    clusters = [json.loads(line) for line in open(clusters_file)]
    saliency = {item["doc_id"]: item for item in [json.loads(line) for line in open(saliency_file)]}

    with open(output_file, "w") as f:
        for doc in clusters:
            if "doc_id" not in doc:
                continue

            sdoc = saliency[doc["doc_id"]]
            salient_spans = set([(span[0], span[1]) for span in sdoc["saliency"] if span[2] == 1])

            salient_clusters = {}
            if "clusters" in doc:
                for cluster, cluster_spans in doc["clusters"].items():
                    cluster_spans = list(map(tuple, cluster_spans))
                    if len(set(cluster_spans) & salient_spans) > 0:
                        salient_clusters[cluster] = cluster_spans
            else:
                for cluster_set in [doc["coref"], doc["coref_non_salient"]]:
                    for cluster, cluster_spans in cluster_set.items():
                        cluster_spans = list(map(tuple, cluster_spans))
                        if len(set(cluster_spans) & salient_spans) > 0:
                            salient_clusters[cluster] = cluster_spans

            if "spans" in doc:
                spans = doc["spans"]
            else:
                spans = [[start, end] for [start, end, _] in doc["ner"]]

            salient_clusters = {k: v for k, v in salient_clusters.items() if len(v) > 0}
            f.write(json.dumps({"doc_id": doc["doc_id"], "clusters": salient_clusters, "spans" : spans}) + "\n")

if __name__ == '__main__' :
    predict(sys.argv[1], sys.argv[2], sys.argv[3])
