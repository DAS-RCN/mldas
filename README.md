MLDAS is a Python package providing tools for studying Distributed Acoustis Sensing (DAS) data and train machine learning algorithms on them. The documentation can be accessed via the following link:

<https://ml4science.gitlab.io/mldas>

# Installation

To install, you can use the Python package manager ``pip`` as follows:

```
sudo pip install mldas
```

Once installed on your system, the package can be loaded into any Python script as follows:

```
from mldas import *
```

```
python mldas/train.py configs/multilabel.yaml -v --depth 2 --lr 0.1 --epochs  2 --sample-size 1 --batch-size 128 --output-dir output_test
```

# Modified BSD License Agreement

MLDAS is released under a modified BSD license. A full description of the license agreement can be found in the [LICENSE.txt](https://gitlab.com/ml4science/mldas/-/blob/master/LICENSE.txt) file.

# About

Machine Learning for Distributed Acoustic Sensing data (MLDAS)
Copyright (c) 2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


