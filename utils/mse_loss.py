#!/usr/bin/env python

def mse(pred,truth):
    try:
        assert len(pred)==len(truth)
        se_seq=[(pred[i]-truth[i])**2 for i in range(len(truth))]
        return sum(se_seq)/len(se_seq)
    
    except IndexError:
        print('the lengths of prediction and truth are not equal!')