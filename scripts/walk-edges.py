# -*- coding:utf-8 -*-
from collections import defaultdict
import cPickle
def add_edge(graph, x, y, num):
    if x not in graph:
        graph[x] = defaultdict()
    graph[x][y] = num

def load_graph(path, undirected=True):
    fr = open(path, 'rb')
    graph = defaultdict()
    for no, line in enumerate(fr):
        x, y = line.strip().split()[:2]
        x = int(x)
        y = int(y)
        add_edge(graph, x, y, no)
        if undirected:
            add_edge(graph, y, x, no)
    return graph

def scan(walks_path, graph):
    node_corpus = []
    edge_corpus = []
    walks = open(walks_path, 'rb')
    for walk in walks:
        nodes_seq = [int(n) for n in walk.strip().split()]
        edges_seq = []
        for i in range(1, len(nodes_seq)):
            u = nodes_seq[i-1]
            v = nodes_seq[i]
            if v not in graph[u]:
                raise ValueError("No edge from %d to %d" % (u, v))
            edges_seq.append(graph[u][v])
        assert len(edges_seq) + 1 == len(nodes_seq)
        node_corpus.append(nodes_seq)
        edge_corpus.append(edges_seq)
    print len(node_corpus), len(edge_corpus)
    cPickle.dump((node_corpus, edge_corpus), open('../citeseer/citeseer.corpus.pkl', 'wb'))
if __name__ == '__main__':
    graph = load_graph('../citeseer/graph.txt')
    scan('../citeseer/citeseer.walks.txt', graph)




