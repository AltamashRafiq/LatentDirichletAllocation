import cython
cimport numpy as np
import numpy as np
from tqdm import tqdm 

@cython.boundscheck(False)
@cython.wraparound(False)
def griffiths_steyvers_cython(long max_iter, double alpha, double beta, long D, long W, long T, long[:] docs_nwords, long[:] all_words, long[:] ta, long[:, :] wt, long[:, :] dt):
    cdef long i, d, w, t0, ta_idx, nwords, wid, t1
    cdef double den_p_dt

    cdef double[:] num_p_dt, unnormalized_p_z, p_z

    cdef double[:] alpha_arr = np.ones(dt.shape[1]) * alpha
    cdef double[:] beta_arr = np.ones(wt.shape[0]) * beta
    cdef double[:] beta_arr2 = np.ones(wt.shape[0]) * beta * W

    for i in tqdm(range(max_iter)):
        ta_idx = 0
        for d in range(D):
            nwords = docs_nwords[d]
            for w in range(nwords):
                # initial topic assignment to token w
                t0 = ta[ta_idx]

                # token id
                wid = all_words[ta_idx]

                # do not include token w (when sampling for token w)
                dt[d, t0] = dt[d, t0] - 1
                wt[t0, wid] = wt[t0, wid] - 1

                num_p_dt = np.asarray(dt[d, :]) + alpha_arr
                den_p_dt = np.sum(dt[d, :]) + (T * alpha)

                num_p_wt = np.asarray(wt[:, wid]) + beta_arr
                den_p_wt = np.asarray(np.sum(wt, axis = 1)) + beta_arr2
        

                unnormalized_p_z = np.abs(np.asarray(num_p_dt) / den_p_dt) * np.abs(np.asarray(num_p_wt) / np.asarray(den_p_wt))
                p_z = np.asarray(unnormalized_p_z) / np.sum(unnormalized_p_z)
                t1 = np.random.choice(np.arange(T), p = np.asarray(p_z))
                ta[ta_idx] = t1

                # re-increment with new topic assignment
                dt[d, t1] = dt[d, t1] + 1
                wt[t1, wid] = wt[t1, wid] + 1
                
                ta_idx += 1
                
    return np.asarray(wt), np.asarray(dt)