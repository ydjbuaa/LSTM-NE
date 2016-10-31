# -*- coding:utf-* -*-
import theano
import theano.tensor as tensor
from theano import config
import numpy

from collections import OrderedDict
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def adadelta(lr, tparams, grads, vars, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(vars, cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

class LSTM(object):
    """
    lstm cell
    """
    def __init__(self, ndim, prefix='lstm'):

        self.dim = ndim
        self.prefix = prefix

        W_values = numpy.concatenate([ortho_weight(ndim),
                           ortho_weight(ndim),
                           ortho_weight(ndim),
                           ortho_weight(ndim)], axis=1)
        self.W = theano.shared(W_values, name=prefix + '_W', borrow=True)

        U_values = numpy.concatenate([ortho_weight(ndim),
                                      ortho_weight(ndim),
                                      ortho_weight(ndim),
                                      ortho_weight(ndim)], axis=1)

        self.U = theano.shared(U_values, name=prefix + '_U', borrow=True)

        b_values = numpy.zeros((4 * ndim,)).astype(config.floatX)

        self.b = theano.shared(b_values, name=prefix + '_b', borrow = True)

        self.params = OrderedDict()
        self.params[prefix+'_W'] = self.W
        self.params[prefix+'_U'] = self.U
        self.params[prefix+'_b'] = self.b


    def output(self, state_below):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(x_, h_, c_):
            preact = tensor.dot(h_, self.U)
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, self.dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, self.dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, self.dim))
            c = tensor.tanh(_slice(preact, 3, self.dim))

            c = f * c_ + i * c

            h = o * tensor.tanh(c)

            return h, c

        state_below = (tensor.dot(state_below, self.W) +self.b)

        rval, updates = theano.scan(_step,
                                    sequences=[state_below],
                                    outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               self.dim),
                                                  tensor.alloc(numpy_floatX(0.),
                                                               n_samples,
                                                               self.dim)],
                                    n_steps= nsteps)
        return rval[0]