import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def rand_choice_nb(arr, p):
    return arr[np.searchsorted(np.cumsum(p), 
                               np.random.random(), 
                               side = "right")]

@njit
def griffiths_steyvers_numba(max_iter, alpha, beta, D, W, T, docs_nwords, all_words, ta, wt, dt):
    for i in range(max_iter):
        ta_idx = 0
        for d in range(D):
            nwords = docs_nwords[d]
            for w in range(nwords):
                t0 = ta[ta_idx] # initial topic assignment to token w
                wid = all_words[ta_idx] # token id

                # do not include token w (when sampling for token w)
                dt[d, t0] = dt[d, t0] - 1
                wt[t0, wid] = wt[t0, wid] - 1

                p_dt = (dt[d, :] + alpha) / (dt[d, :].sum() + T * alpha)
                p_wt = (wt[:, wid] + beta) / (wt.sum(axis = 1) + W * beta)

                unnormalized_p_z = p_dt * p_wt
                p_z = unnormalized_p_z / unnormalized_p_z.sum()

                # draw topic for word n from multinomial using probabilities calculated above
                t1 = rand_choice_nb(np.arange(T), p_z)
                ta[ta_idx] = t1

                # re-increment with new topic assignment
                dt[d, t1] = dt[d, t1] + 1
                wt[t1, wid] = wt[t1, wid] + 1
                
                ta_idx += 1
                
    return wt, dt