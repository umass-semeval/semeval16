class ConfusionMatrix(object):

    def __init__(self, y, ypred, classnames):
        self.y = y
        self.ypred = ypred
        self.classnames = classnames
        self.tp = None
        self.fp = None
        self.fn = None
        self.true_counts = None
        self.pred_counts = None
        self.ncorrect = 0.0
        self.total = 0.0
        self._build()

    def _build(self):
        classnames = self.classnames
        true_counts = {x: 0 for x in classnames}
        pred_counts = {x: 0 for x in classnames}
        tp = {x: 0 for x in classnames}
        fp = {x: 0 for x in classnames}
        fn = {x: 0 for x in classnames}
        ncorrect = 0.0
        total = 0.0
        for i, ipred in zip(self.y, self.ypred):
            true_classname = classnames[i]
            true_counts[true_classname] += 1
            pred_classname = classnames[ipred]
            pred_counts[pred_classname] += 1
            if i != ipred:
                fp[pred_classname] += 1
                fn[true_classname] += 1
            else:
                ncorrect += 1.0
                tp[true_classname] += 1
            total += 1.0
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.true_counts = true_counts
        self.pred_counts = pred_counts
        self.ncorrect += ncorrect
        self.total += total

    def recall(self, classname):
        tp, fp, fn = 1.*self.tp[classname], 1.*self.fp[classname], 1.*self.fn[classname]
        if (tp+fn) == 0:
            return 0.0
        else:
            return tp/(tp+fn)

    def precision(self, classname):
        tp, fp, fn = 1.*self.tp[classname], 1.*self.fp[classname], 1.*self.fn[classname]
        if (tp+fp) == 0:
            return 0.0
        else:
            return tp/(tp+fp)

    def p_r_f1(self, classname):
        p = self.precision(classname)
        r = self.recall(classname)
        if (p+r) == 0:
            return p, r, 0.0
        else:
            return p, r, 2.*p*r/(p+r)

    def __repr__(self):
        table = {x: [] for x in self.classnames}
        overall_f1, overall_p, overall_r = 0.0, 0.0, 0.0
        for c in self.classnames:
            p, r, f1 = self.p_r_f1(c)
            table[c] = [f1, p, r]
            overall_p += p
            overall_r += r
            overall_f1 += f1
        n = 1. * len(self.classnames)
        avg_f1 = overall_f1/n
        avg_p = overall_p/n
        avg_r = overall_r/n
        acc = (self.ncorrect / self.total)*100.
        string = "%s:\tf1=%f\tp=%f\tr=%f (acc=%f %d/%d)\n" % \
                 ("overall", avg_f1, avg_p, avg_r, acc, int(self.ncorrect), int(self.total))
        for c, vals in table.items():
            f1,p,r = vals[0], vals[1], vals[2]
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            true_count = self.true_counts[c]
            pred_count = self.pred_counts[c]
            string += "%s:\tf1=%f\tp=%f\tr=%f (tp=%d, fp=%d, fn=%d, true=%d, pred=%d)\n" % \
                      (c, f1, p, r, tp, fp, fn, true_count, pred_count)
        return string




