import numpy as np
import cPickle

## AutoEncoder Class Example
class AutoEncoder(object):
    def __init__(self, nvis, nhid, iscale = 0.1,
                    activation_fn = np.tanh,
                    params = None):
        self.nvis = nvis
        self.nhid = nhid
        self.activation_fn = activation_fn
        if params is None:
            self.W = iscale * np.random.randn(nvis, nhid)
            self.bias_vis = np.zeros(nvis)
            self.bias_hid = np.zeros(nhid)
        else:
            self.W, self.bias_vis, self.bias_hid = params
        print self
    def __str__(self):
        model_str = '%s\n' % self.__class__.__name__
        model_str += '\tnvis = %i\n' % self.nvis
        model_str += '\tnhid = %i\n' % self.nhid
        model_str += '\tactivation_fn = %s\n' % str(self.activation_fn)
        model_str += '\tmean std(weigths) = %.2f\n' % self.W.std(axis=0).mean()
        return model_str
    def save(self, fname):
        with open(fname, 'w') as f:
            cPickle.dump([self.W, self.bias_vis, self.bias_hid], f)