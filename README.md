# pyCLiMF

Collaborative Less is More Matrix Factorization in Python

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic ACM RecSys 2012

Basically, this is a cythonized version of https://github.com/gamboviol/climf/

## Installation
```
python setup.py build_ext --inplace
```

## Getting Started

Here is an example run on the epinions dataset:

```python
from scipy.io.mmio import mmread
from climf import CLiMF
import numpy as np

data = mmread("EP25_UPL5_train.mtx").tocsr()
testdata = mmread("EP25_UPL5_test.mtx").tocsr()

cf = CLiMF(lbda=0.001, gamma=0.0001, dim=10, max_iters=10)

cf.fit(data)
iteration 1:
train mrr = 0.05750323
iteration 2:
train mrr = 0.06994045
iteration 3:
train mrr = 0.07469046
iteration 4:
train mrr = 0.07611490
iteration 5:
train mrr = 0.07690448
iteration 6:
train mrr = 0.07711117
iteration 7:
train mrr = 0.07657648
iteration 8:
train mrr = 0.07625065
iteration 9:
train mrr = 0.07590930
iteration 10:
train mrr = 0.07558414

print "Test MRR: %.8f" % cf.compute_mrr(testdata)
Test MRR: Test MRR: 0.40167283
```
