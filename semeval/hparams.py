class HParams(object):
    def __init__(self,
                 nepochs=30,
                 batchsize=512,
                 learning_rate=0.1,
                 bidirectional=True,
                 nhidden=256,
                 embedding_dim=50,
                 pool='mean',
                 grad_clip=100,
                 optimizer='adam'):
        
        self.nepochs = nepochs
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.bidirectional = bool(bidirectional)
        self.nhidden = nhidden
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.grad_clip = grad_clip
        self.optimizer = optimizer
    
    def parse_args(self, args):        
        self.nepochs = args.nepochs
        self.batchsize = args.batchsize
        self.learning_rate = args.learning_rate
        self.bidirectional = bool(args.bidirectional)
        self.nhidden = args.nhidden
        self.embedding_dim = args.embedding_dim
        self.pool = args.pool
        self.grad_clip = args.grad_clip
        self.optimizer = args.optimizer
        
    def __repr__(self):
        hps = {}
        hps['nepochs'] = self.nepochs
        hps['batchsize'] = self.batchsize
        hps['learning_rate'] = self.learning_rate
        hps['bidirectional'] = self.bidirectional
        hps['nhidden'] = self.nhidden
        hps['embedding_dim'] = self.embedding_dim
        hps['pool'] = self.pool
        hps['grad_clip'] = self.grad_clip
        hps['optimizer'] = self.optimizer
        srt = sorted(hps.items(), key=lambda x: x[0])
        head = '========== HPARAMS ========='
        string = head + '\n'
        for k, v in srt:
            string += '%s: %s\n' % (k, v)
        end = ''.join(['=']*len(head))
        string += end + '\n'
        return string
