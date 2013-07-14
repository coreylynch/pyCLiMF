pyCLiMF
=======

Collaborative Less is More Matrix Factorization in Python

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic ACM RecSys 2012

Basically, this is a cythonized version of https://github.com/gamboviol/climf/

Here is an example run on the epinions dataset:

```python
from scipy.io.mmio import mmread
from climf import CLiMF
import numpy as np

data = mmread("EP25_UPL5_train.mtx").tocsr()
testdata = mmread("EP25_UPL5_test.mtx").tocsr()

cf = CLiMF()
cf.fit(data)
cf.compute_mrr(testdata)

0.39684884842965751
```
