# -*- coding:utf-8 -*-
import numpy
import theano.tensor
from collections import OrderedDict
import cPickle
import time
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
    y = tensor.matrix('nodes', dtype='int64')

    n_steps = x.shape[0]
    n_samples = x.shape[1]

    params = OrderedDict()

    # add embs to params
    Nembs, Eembs = load_embeddings(options['n_nodes'], options['n_edges'], options['dim'])
    params['Nembs'] = theano.shared(Nembs, name='Nembs')
    params['Eembs'] = theano.shared(Eembs, name='Eembs')

    x_emb = params['Nembs'][x.flatten()].reshape([n_steps, n_samples, options['dim']])
    y_emb = params['Eembs'][y.flatten()].reshape([n_steps, n_samples, options['dim']])

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

    ne_err = tensor.mean((xn_proj - y_emb) ** 2)
    ee_err = tensor.mean((xe_proj- x_emb[1:-1]) **2)

    cost = ne_err + ee_err

    return use_noise, x, y, params, cost

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_dataset():
    node_corpus, edge_corpus, n_nodes, n_edges = cPickle.load(open('./citeseer/citeseer.corpus.pkl', 'rb'))
    return node_corpus, edge_corpus, n_nodes, n_edges
    pass

def prepare_data(n_seqs, e_seqs, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in n_seqs]

    if maxlen is not None:
        new_n_seqs = []
        new_e_seqs = []
        new_lengths = []
        for l, n, e in zip(lengths, n_seqs, e_seqs):
            if l < maxlen:
                new_n_seqs.append(n)
                new_e_seqs.append(e)
                new_lengths.append(l)

        lengths = new_lengths
        e_seqs = new_e_seqs
        n_seqs = new_n_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(n_seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    y = numpy.zeros((maxlen, n_samples)).astype('int64')

    for idx, s in enumerate(n_seqs):
        x[:lengths[idx]-2, idx] = s[1:-1]
    for idx, e in enumerate(e_seqs):
        y[:lengths[idx]-1, idx] = e

    return x, y

def train(
        dim = 128,
        batch_size=64,
        use_dropout=True,
        max_epochs = 1,
        lrate=0.0001,
        dispFreq = 10,
    ):
    options = locals().copy()
    print 'Model Options:', options

    node_corpus, edge_corpus, n_nodes, n_edges = get_dataset()
    options['n_nodes'] = n_nodes
    options['n_edges'] = n_edges

    print 'Building Model'
    use_noise, x, y, params, cost = build_model(options)

    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(params.values()))
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, params, grads,
                                        [x, y], cost)
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(node_corpus), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [node_corpus[t] for t in train_index]
                x = [edge_corpus[t] for t in train_index]

                y, x = prepare_data(y, x)
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, y)
                f_update(lrate)

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

    except KeyboardInterrupt:
        print("Training interupted")

if __name__ == '__main__':
    train()




