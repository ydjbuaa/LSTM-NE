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
    n_nodes = 0
    n_edges = 0
    for no, line in enumerate(fr):
        x, y = line.strip().split()[:2]
        x = int(x)
        y = int(y)
        add_edge(graph, x, y, no)
        n_edges += 1
        if undirected:
            add_edge(graph, y, x, no)
    n_nodes = len(graph.keys())
    return graph, n_nodes, n_edges

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

    return node_corpus, edge_corpus


if __name__ == '__main__':
    graph, n_nodes, n_edges = load_graph('../citeseer/graph.txt')
    print n_nodes, n_edges
    node_corpus,edge_corpus = scan('../citeseer/citeseer.walks.txt', graph)
    print len(node_corpus), len(edge_corpus)
    cPickle.dump((node_corpus, edge_corpus, n_nodes, n_edges), open('../citeseer/citeseer.corpus.pkl', 'wb'))



