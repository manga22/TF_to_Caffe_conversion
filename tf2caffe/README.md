# Porting a Tensorflow model to Caffe

This document provides with instructions to install Tensorflow, Caffe and porting a model from tensorflow to caffe.
The following assumes that Python 3.4 is set up and executable on your Ubuntu 14.04 system.
    
# Installation:

1. Tensor Flow 
 
   - Launch the terminal and execute the following command to install Tensor Flow with CPU support [https://www.tensorflow.org/install/install_linux]
	
	$sudo pip3 install --upgrade \
 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp34-cp34m-linux_x86_64.whl
	
2. Caffe

   - Installing General dependencies

	$ sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev 
	$ sudo apt-get install --no-install-recommends libboost-all-dev
	$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
	
   - Install protobuf 3.4 compiler as python 3.4 is used 

	- Download the required protobuf package corresponding to the python version from [https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-python-3.4.1.zip]
	- Unzip the package and execute the following commands to install the Protocol Buffer compiler

		 $ cd path/to/protobuf-python-3.4.1
		 $ ./configure
		 $ make
		 $ make check
		 $ sudo make install
		 $ sudo ldconfig /usr/local/lib

   - Building Caffe[http://caffe.berkeleyvision.org/install_apt.html][https://prateekvjoshi.com/2016/01/05/how-to-install-caffe-on-ubuntu/]

	$ git clone https://github.com/BVLC/caffe.git
	$ cd caffe
	$ cp Makefile.config.example Makefile.config   #make changes in the Makefile.config to use CPU only and python3
	$ make all -j4
	$ make test 

	#Reboot your machine at this point and run tests to check if caffe is installed properly

	$ sudo reboot
	$ cd /path/to/downloaded/caffe
	$ sudo make runtest

	#Install python interface for caffe

	$ sudo make pycaffe

3. OpenCV-3.2

   - Download the installation script [https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh] and run the following

	$ sudo bash install-opencv.sh
	 

# Points to be noted while porting a Tensor Flow model to Caffe

   - While porting the tensorflow model parameters to caffe parameters, re-order the tf kernel with shape [h, w, in, out] to caffe kernel with shape [out, in, h, w]. This can be done using:
	kernelcaffe=np.transpose(kerneltf,[3,2,0,1]) 
   - To read tensors from a tensorflow checkpoint file, refer [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py]
   - The model parameters are named as:
	Tensorflow:
		Weights: layer_name/kernels
		Biases : layer_name/biases
	Caffe:
		Weights: layer_name[0]
		Biases : layer_name[1] 

   - Operations like tf.reshape(),tf.transpose(), etc. on tensors can be replaced by numpy operations in a caffe model. 
   - The shapes of tensors(in tensorflow) and blobs(in caffe) should be taken care of while porting.
         	


 


