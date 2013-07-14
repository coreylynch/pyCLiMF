# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Corey Lynch <coreylynch9@gmail.com>
#

import numpy as np
import scipy 
from libc.math cimport exp, log, pow
cimport numpy as np
cimport cython

np.import_array()

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INTEGER

cdef g(DOUBLE x):
    """sigmoid function"""
    return 1/(1+exp(-x))

cdef dg(DOUBLE x):
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

cdef np.ndarray[DOUBLE, ndim=1, mode='c'] precompute_f(np.ndarray[DOUBLE, ndim=2, mode='c'] U,
                                                       np.ndarray[DOUBLE, ndim=2, mode='c'] V,
                                                       INTEGER *x_ind_ptr,
                                                       int xnnz,
                                                       int num_factors,
                                                       int i):
    """precompute f[j] = <U[i],V[j]>
        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          i   : user of interest
        returns:
          dot products <U[i],V[j]> for all j in data[i]
    """
    cdef unsigned int j = 0
    cdef unsigned int factor = 0
    cdef DOUBLE dot_prod

    # create f as a ndarray of len(nnz)
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] f = np.zeros(xnnz, dtype=np.float64, order="c")
    for j in range(xnnz):
        dot_prod = 0.0
        for factor in range(num_factors):
            dot_prod += U[i,factor] * V[x_ind_ptr[j],factor]
        f[j] = dot_prod
    return f

def compute_mrr_fast(np.ndarray[int, ndim=1, mode='c'] test_user_ids, np.ndarray test_user_data,
                np.ndarray[DOUBLE, ndim=2, mode='c'] U, np.ndarray[DOUBLE, ndim=2, mode='c'] V):
    
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] mrr = np.zeros(test_user_ids.shape[0], dtype=np.float64, order="c")
    cdef unsigned int ix
    cdef unsigned int i
    cdef unsigned int item
    cdef unsigned int item_idx
    cdef unsigned int user_idx
    cdef set items
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] predictions = np.zeros(V.shape[0], dtype=np.float64, order="c")
    cdef DOUBLE pred
    cdef int rank
    cdef int num_factors = U.shape[1]
    cdef np.ndarray[INTEGER, ndim=1, mode='c'] test_user

    for i in range(test_user_ids.shape[0]):
        test_user = test_user_data[i]
        items = {item for item in test_user}
        for item_idx in range(V.shape[0]):
            pred = 0.0
            for factor in range(num_factors):
                user_idx = test_user_ids[i]
                pred += U[user_idx, factor] * V[item_idx, factor]
            predictions[item_idx] = pred
    
        ranked_preds = np.argsort(predictions)
        ranked_preds= ranked_preds[::-1]
        for rank,item in enumerate(ranked_preds):
            if item in items:
                mrr[i] = 1.0/(rank+1)
                break
    assert(len(mrr) == len(test_user_ids))
    return np.mean(mrr)

def climf_fast(CSRDataset dataset,
               np.ndarray[DOUBLE, ndim=2, mode='c'] U,
               np.ndarray[DOUBLE, ndim=2, mode='c'] V,
               double lbda,
               double gamma,
               int n_factors,
               int n_iter,
               int shuffle,
               int seed,
               np.ndarray[int, ndim=1, mode='c'] sample_user_ids,
               np.ndarray sample_user_data):

    # get the data information into easy vars
    cdef Py_ssize_t n_samples = dataset.n_samples
    cdef Py_ssize_t n_users = U.shape[0]
    cdef Py_ssize_t n_items = V.shape[0]

    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] f

    cdef DOUBLE * x_data_ptr = NULL
    cdef INTEGER * x_ind_ptr = NULL

    # helper variable
    cdef int xnnz
    cdef double eta = 0.0
    cdef double p = 0.0
    cdef DOUBLE y = 0.0
    cdef DOUBLE sample_weight = 1.0
    cdef unsigned int i = 0
    cdef unsigned int t = 0
    cdef unsigned int j = 0
    cdef unsigned int k = 0
    cdef unsigned int idx = 0
    cdef unsigned int idx_j = 0
    cdef unsigned int idx_k = 0
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] dU = np.zeros(n_factors, dtype=np.float64, order="c")
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] dV = np.zeros(n_factors, dtype=np.float64, order="c")
    cdef DOUBLE dVUpdate = 0.0
    cdef DOUBLE dUUpdate = 0.0
    cdef np.ndarray[DOUBLE, ndim=1, mode='c'] V_j_minus_V_k = np.zeros(n_factors, dtype=np.float64, order="c")

    for t in range(n_iter):
        if shuffle > 0:
            dataset.shuffle(seed)

        for i in range(n_users):
            dataset.next( & x_data_ptr, & x_ind_ptr, & xnnz, & y,
                             & sample_weight)

            # dU = -lbda * U[i]
            for idx in range(n_factors):
                dU[idx] = -lbda * U[i, idx]

            f = precompute_f(U, V, x_ind_ptr, xnnz, n_factors, i)

            for j in range(xnnz):
                idx_j = x_ind_ptr[j]
                # dV = g(-f[j])-lbda*V[j]
                for idx in range(n_factors):
                     dV[idx] = g(-f[j]) - lbda * V[idx_j, idx]

                for k in range(xnnz):
                    dVUpdate = dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))
                    # dV += dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))*U[i]
                    for idx in range(n_factors):
                        dV[idx] += dVUpdate * U[i, idx]
                    
                # V[j] += gamma*dV
                for idx in range(n_factors):
                    V[idx_j, idx] += gamma * dV[idx]
                # dU += g(-f[j])*V[idx_j]
                dUUpdate = g(-f[j])
                for idx in range(n_factors):
                    dU[idx] += dUUpdate * V[idx_j, idx]
                for k in range(xnnz):
                    idx_k = x_ind_ptr[k]
                    # dU += (V[j]-V[k])*dg(f[k]-f[j])/(1-g(f[k]-f[j]))
                    for idx in range(n_factors):
                        V_j_minus_V_k[idx] = V[idx_j, idx] - V[idx_k, idx]
                    for idx in range(n_factors):
                        dU[idx] += V_j_minus_V_k[idx] * dg(f[k]-f[j])/(1-g(f[k]-f[j]))
            # U[i] += gamma*dU
            for idx in range(n_factors):
                U[i, idx] += gamma * dU[idx]

        print 'iteration {0}:'.format(t+1)
        print 'train mrr = {0:.8f}'.format(compute_mrr_fast(sample_user_ids, sample_user_data, U, V))


cdef class CSRDataset:
    """An sklearn ``SequentialDataset`` backed by a scipy sparse CSR matrix. This is an ugly hack for the moment until I find the best way to link to sklearn. """

    cdef Py_ssize_t n_samples
    cdef int current_index
    cdef int stride
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr
    cdef np.ndarray feature_indices
    cdef INTEGER *feature_indices_ptr
    cdef np.ndarray index
    cdef INTEGER *index_data_ptr
    cdef DOUBLE *sample_weight_data

    def __cinit__(self, np.ndarray[DOUBLE, ndim=1, mode='c'] X_data,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indptr,
                  np.ndarray[INTEGER, ndim=1, mode='c'] X_indices,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] Y,
                  np.ndarray[DOUBLE, ndim=1, mode='c'] sample_weight):
        """Dataset backed by a scipy sparse CSR matrix.

        The feature indices of ``x`` are given by x_ind_ptr[0:nnz].
        The corresponding feature values are given by
        x_data_ptr[0:nnz].

        Parameters
        ----------
        X_data : ndarray, dtype=np.float64, ndim=1, mode='c'
            The data array of the CSR matrix; a one-dimensional c-continuous
            numpy array of dtype np.float64.
        X_indptr : ndarray, dtype=np.int32, ndim=1, mode='c'
            The index pointer array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        X_indices : ndarray, dtype=np.int32, ndim=1, mode='c'
            The column indices array of the CSR matrix; a one-dimensional
            c-continuous numpy array of dtype np.int32.
        Y : ndarray, dtype=np.float64, ndim=1, mode='c'
            The target values; a one-dimensional c-continuous numpy array of
            dtype np.float64.
        sample_weights : ndarray, dtype=np.float64, ndim=1, mode='c'
            The weight of each sample; a one-dimensional c-continuous numpy
            array of dtype np.float64.
        """
        self.n_samples = Y.shape[0]
        self.current_index = -1
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data
        self.sample_weight_data = <DOUBLE *> sample_weight.data
        # Use index array for fast shuffling
        cdef np.ndarray[INTEGER, ndim=1,
                        mode='c'] index = np.arange(0, self.n_samples,
                                                    dtype=np.int32)
        self.index = index
        self.index_data_ptr = <INTEGER *> index.data

    cdef void next(self, DOUBLE **x_data_ptr, INTEGER **x_ind_ptr,
                   int *nnz, DOUBLE *y, DOUBLE *sample_weight):
        cdef int current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        cdef int sample_idx = self.index_data_ptr[current_index]
        cdef int offset = self.X_indptr_ptr[sample_idx]
        y[0] = self.Y_data_ptr[sample_idx]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        nnz[0] = self.X_indptr_ptr[sample_idx + 1] - offset
        sample_weight[0] = self.sample_weight_data[sample_idx]

        self.current_index = current_index

    cdef void shuffle(self, seed):
        np.random.RandomState(seed).shuffle(self.index)

