################################
pl-image_classification
################################


Abstract
********

This is a demo ChRIS plugin app.

The image plugin runs a neural network to perform a supervised machine learning 
task (Classification) on the MNSIT data set.

The MNIST data set is a database of images of handwriten digits, with 60,000 
training data points and 10,000 test data points.

You can learn more about the MNIST data set here: http://yann.lecun.com/exdb/mnist/

Run
***

* In your pwd, make two directories and name them 'in' and 'out'.

* Download the dataset from http://yann.lecun.com/exdb/mnist/ and move the 
downloaded files (.gz) files to the 'in' folder.


* Using ``docker run``
====================

Assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``

.. code-block:: bash

    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing   \
            fnndsc/pl-image_classification image_classification.py            \
            /incoming /outgoing

This will ...

Make sure that the host ``$(pwd)/out`` directory is world writable!







