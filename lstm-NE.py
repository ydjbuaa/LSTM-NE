# -*- coding:utf-8 -*-
import numpy
import theano.tensor
from collections import OrderedDict

from lstm import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def load_embeddings(n_nodes, n_edges, dim=128):
    randn = numpy.random.rand(n_nodes, dim)
    node_embs = (randn * 0.01).astype(config.floatX)

    randn = numpy.random.rand(n_edges, dim)
    edge_embs = (randn * 0.01).astype(config.floatX)

    return node_embs, edge_embs

def build_model(options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('edges', dtype='int64')
    y = tensor.vector('nodes', dtype='int64')

    n_steps = x.shape[0]
    n_samples = x.shape[1]

    params = OrderedDict()

    # add embs to params
    Nembs, Eembs = load_embeddings(options['n_nodes'], options['n_edges'], options['dim'])
    params['Nembs'] = theano.shared(Nembs, name='Nembs')
    params['Eembs'] = theano.shared(Eembs, name='Eembs')

    x_emb = params['Nembs'][x.flatten()].reshape(n_steps, n_samples, options['dim'])
    y_emb = params['Eembs'[y.flatten()]].reshape(n_steps, n_samples, options['dim'])

    # build edge-to-nodes LSTM
    e2n_lstm = LSTM(options['dim'], prefix='e2n_lstm')
    # add lstm params
    for kk, pp in e2n_lstm.params.items():
        params[kk] = pp

    xn_proj = e2n_lstm.output(x_emb)
    xn_proj = xn_proj[1:]

    n2e_lstm = LSTM(options['dim'], prefix='n2e_lstm')
    for kk, pp in n2e_lstm.params.items():
        params[kk] = pp

    xe_proj = n2e_lstm.output(xn_proj)
    xe_proj = xe_proj[1:]

    ne_err = ((xn_proj - y_emb) ** 2).sum()
    ee_err = ((xe_proj- x_emb[1:-1]) **2).sum()



