ERAN <img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg">
========

![High Level](https://raw.githubusercontent.com/eth-sri/diffai/master/media/overview.png)

ETH Robustness Analyzer for Neural Networks (ERAN) is a state-of-the-art sound, precise, and scalable analyzer based on [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation).  ERAN is developed at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Safe AI project](http://safeai.ethz.ch/). The goal of ERAN is to automatically verify robustness of neural networks with feedforward, convolutional, and residual layers against input perturbations (e.g.,  L∞-norm attacks, geometric transformations, etc). 

ERAN supports networks with ReLU, Sigmoid and Tanh activations and is sound under floating point arithmetic. It employs custom abstract domains which are specifically designed for the setting of neural networks and which aim to balance scalability and precision. Specifically, ERAN is based on two abstract domains:

* DeepZ [NIPS'18]: Zonotope domain containing specialized abstract transformers for handling ReLU, Sigmoid and Tanh activation functions.

* DeepPoly [POPL'19]: A domain that combines floating point Polyhedra with Intervals.

Both domains are implemented in the [ELINA](http://elina.ethz.ch/) library for numerical abstractions. More details can be found in the publications below.

Requirements 
------------
GNU C compiler, ELINA

python3.6 or higher, tensorflow 1.11 or higher, numpy.


Installation
------------
Clone the ERAN repository via git as follows:
```
git clone https://github.com/eth-sri/ERAN.git
cd ERAN
```

The dependencies for ERAN can be installed step by step as follows (sudo rights might be required):

Install m4:
```
wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm m4-1.4.1.tar.gz
```

Install gmp:
```
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz
```

Install mpfr:
```
wget https://www.mpfr.org/mpfr-current/mpfr-4.0.1.tar.xz
tar -xvf mpfr-4.0.1.tar.xz
cd mpfr-4.0.1
./configure
make
make install
cd ..
rm mpfr-4.0.1.tar.xz
```

Install ELINA:
```
git clone https://github.com/eth-sri/ELINA.git
cd ELINA
./configure
make
make install
```

Update the LD_LIBRARY_PATH:
```
export LD_BLIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

We also provide an install script that will install ELINA and all the necessary dependencies. One can run it as follows:

```
sudo./install.sh

```

To install the python dependencies (numpy and tensorflow), type:

```
pip3 install -r requirements.txt
```

ERAN may not be compatible with older versions of tensorflow (we have tested ERAN with versions >= 1.11.0), so if you have an older version and want to keep it, then we recommend using the python virtual environment for installing tensorflow.


Usage
-------------

```
cd tf-verify

python3 . <netname> <epsilon> <domain> <dataset>
```

* <netname>: path to the network file.

* <epsilon>: A float value between 0 and 1 specifying bound for the L∞-norm based perturbation.

* <domain>: can be either 'deepzono' (for the DeepZ domain) or 'deeppoly'. Note that the residual layers are currently only supported with the DeepZ domain. 

* <dataset>: can be either 'mnist' or 'cifar10'

Example
-------------

```
python3 . ../nets/pytorch/mnist/convBig__DiffAI.pyt 0.1 deepzono mnist
```

will evaluate the local robustness of the MNIST convolutional network (upto 35K neurons) with ReLU activation trained using DiffAI on the 100 MNIST test images. In the above setting, epsilon=0.1 and the domain used by our analyzer is the deepzono domain. Our analyzer will print the following:

* 'Verified' for an image when it can prove the robustness of the network and 'Failed' when it cannot. It will also print an error message when the network misclassifies an image.

* the timing in seconds.

* The ratio of images on which the network is robust versus the number of images on which it classifies correctly.
 


Publications
-------------

*  [Fast and Effective Robustness Certification](https://www.sri.inf.ethz.ch/publications/singh2018effective). 

    Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Püschel, and Martin Vechev. 

    NIPS 2018.



*  [An Abstract Domain for Certifying Neural Networks](https://www.sri.inf.ethz.ch/publications/singh2019domain).

    Gagandeep Singh, Timon Gehr, Markus Püschel, and Martin Vechev. 

    POPL 2019.


Neural Networks and Datasets
---------------

We provide a number of pretrained MNIST and CIAFR10 defended and undefended feedforward and convolutional neural networks with ReLU, Sigmoid and Tanh activations trained with the PyTorch and TensorFlow frameworks. The adversarial training to obtain the defended networks is performed using PGD and [DiffAI](https://github.com/eth-sri/diffai). 

| Dataset  |   Model  |  Type   | #units | #layers| Activation | Training Defense| Download |
| :-------- | :-------- | :-------- | :-------------| :-------------| :------------ | :------------- | :---------------:|
| MNIST   | 3x100   | fully connected | 310 | 3    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf)|
|         | 6x100   | fully connected | 610 | 6    | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_100.tf)|
|         | 9x200 | fully connected | 1,810 | 9   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf)|
|         | 6x500 | fully connected | 3,010 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | ReLU  | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 |  6  | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.3_6_500.pyt)|
|         | 6x500 | fully connected | 3,010  | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 |  6  | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.3_6_500.pyt)|
|         | 6x500  | fully connected | 3,010 | 6 |    Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 |  fully connected| 3,010 | 6   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.1_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.3_6_500.pyt)|
|         | 4x1024 | fully connected | 4,106 | 4   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_4_1024.tf)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__Point.pyt)|
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__PGDK.pyt) |
|         |  ConvSmall | convolutional | 3,604 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__DiffAI.pyt) |
|         | ConvMed | convolutional | 4,804 | 3  | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 4,804 | 3   | ReLU | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.1.pyt) |
|         | ConvMed | convolutional | 4,804 | 3   | ReLU | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.3.pyt) |
|         | ConvMed | convolutional | 4,804 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 4,804 | 3   | Sigmoid | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 4,804 | 3   | Sigmoid | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.3.pyt) | 
|         | ConvMed | convolutional | 4,804 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 4,804 | 3   | Tanh | PGD &epsilon;=0.1 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.1.pyt) | 
|         | ConvMed | convolutional | 4,804 | 3   |  Tanh | PGD &epsilon;=0.3 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.3.pyt) |
|         | ConvMaxpool | convolutional | 13,798 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_conv_maxpool.tf)|
|         | ConvBig | convolutional | 34,688 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convBigRELU__DiffAI.pyt) |
|         | ConvSuper | convolutional | 88,500 | 6  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSuperRELU__DiffAI.pyt) |
|         | Skip      | Residual | 71,650 | 9 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/skip__DiffAI.pyt) |
| CIFAR10 | 4x100 | fully connected | 410 | 4 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_4_100.tf) |
|         | 6x100 | fully connected | 610 | 6 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_6_100.tf) |
|         | 9x200 | fully connected | 1,810 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf) |
|         | 6x500 | fully connected | 3,010 | 6   | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,010 | 6   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0313_6_500.pyt)| 
|         | 6x500 | fully connected | 3,010 | 6   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__Point_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0078_6_500.pyt)|
|         | 6x500 | fully connected | 3,010 | 6   | Tanh | PGD &epsilon;=0.0313 |  [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0313_6_500.pyt)| 
|         | 7x1024 | fully connected | 7,178 | 7 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_7_1024.tf) |
|         | ConvSmall | convolutional | 4,852 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__Point.pyt)|
|         | ConvSmall   | convolutional  | 4,852 | 3  | ReLU  | PGD | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__PGDK.pyt)|
|         | ConvSmall  | convolutional | 4,852 | 3  | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__DiffAI.pyt)|
|         | ConvMed | convolutional | 6,244 | 3 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__Point.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | ReLU | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | ReLU | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 6,244 | 3   | Sigmoid | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__Point.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | Sigmoid | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | Sigmoid | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0313.pyt) | 
|         | ConvMed | convolutional | 6,244 | 3   | Tanh | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__Point.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | Tanh | PGD &epsilon;=0.0078 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0078.pyt) |
|         | ConvMed | convolutional | 6,244 | 3   | Tanh | PGD &epsilon;=0.0313 | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0313.pyt) |  
|         | ConvMaxpool | convolutional | 53,938 | 9 | ReLU | None | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_conv_maxpool.tf)|
|         | ConvBig | convolutional | 62,464 | 6 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt) | 

We provide the first 100 images from the testset of both MNIST and CIFAR10 datasets in the 'data' folder. Our analyzer first verifies whether the neural network classifies an image correctly before performing robustness analysis.

Experimental Results
--------------
We ran our experiments for the feedforward networks on a 3.3 GHz 10 core Intel i9-7900X Skylake CPU with a main memory of 64 GB whereas our experiments for the convolutional networks were run on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with 512 GB of main memory. We first compare the precision and performance of DeepZ and DeepPoly vs [Fast-Lin](https://github.com/huanzhang12/CertifiedReLURobustness) on the MNIST 6x100 network in single threaded mode. It can be seen that DeepZ has the same precision as Fast-Lin whereas DeepPoly is more precise while also being faster.

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_6_100.png)

In the following, we compare the precision and performance of DeepZ and DeepPoly on a subset of the neural networks listed above in multi-threaded mode. In can be seen that DeepPoly is overall more precise than DeepZ but it is slower than DeepZ on the convolutional networks. 

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_6_500.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_convsmall.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/mnist_sigmoid_tanh.png)

![High Level](https://files.sri.inf.ethz.ch/eran/plots/cifar10_convsmall.png)


The table below compares the performance and precision of DeepZ and DeepPoly on our large networks trained with DiffAI. 


<table aligh="center">
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>&epsilon;</td>
    <td colspan="2">% Verified Robustness</td>
    <td colspan="2">% Average Runtime (s)</td>
  </tr>
  <tr>
   <td> </td>
   <td> </td>
   <td> </td>
   <td> DeepZ </td>
   <td> DeepPoly </td>
   <td> DeepZ </td> 
   <td> DeepPoly </td>
  </tr>

<tr>
   <td> MNIST</td>
   <td> ConvBig</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 5 </td> 
   <td> 50 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.2</td>
   <td> 79 </td>
   <td> 78 </td>
   <td> 7 </td> 
   <td> 61 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.3</td>
   <td> 37 </td>
   <td> 43 </td>
   <td> 17 </td> 
   <td> 88 </td>
</tr>

<tr>
   <td> </td>
   <td> ConvSuper</td>
   <td> 0.1</td>
   <td> 97 </td>
   <td> 97 </td>
   <td> 133 </td> 
   <td> 400 </td>
</tr>

<tr>
   <td> </td>
   <td> Skip</td>
   <td> 0.1</td>
   <td> 95 </td>
   <td> N/A </td>
   <td> 29 </td> 
   <td> N/A </td>
</tr>

<tr>
   <td> CIFAR10</td>
   <td> ConvBig</td>
   <td> 0.006</td>
   <td> 50 </td>
   <td> 52 </td>
   <td> 39 </td> 
   <td> 322 </td>
</tr>


<tr>
   <td> </td>
   <td> ConvBig</td>
   <td> 0.008</td>
   <td> 33 </td>
   <td> 40 </td>
   <td> 46 </td> 
   <td> 331 </td>
</tr>


</table>

More experimental results can be found in our papers.

Contributors
--------------

* [Gagandeep Singh](https://www.sri.inf.ethz.ch/people/gagandeep) - gsingh@inf.ethz.ch

* [Matthew Mirman](https://www.mirman.com) - matt@mirman.com

* [Timon Gehr](https://www.sri.inf.ethz.ch/tg.php) - timon.gehr@inf.ethz.ch
 
* Adrian Hoffmann - adriahof@student.ethz.ch

* [Markus Püschel](https://acl.inf.ethz.ch/people/markusp/) - pueschel@inf.ethz.ch

* [Martin Vechev](https://www.sri.inf.ethz.ch/vechev.php) - martin.vechev@inf.ethz.ch

License and Copyright
---------------------

* Copyright (c) 2018 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)
