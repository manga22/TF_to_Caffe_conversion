from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import tensorflow as tf
import caffe
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None

caffe.set_mode_cpu()
net = caffe.Net('SqueezeDet.prototxt', caffe.TEST)
tf_checkpoint_filename='model.ckpt-87000'
all_tensors=True
with tf.Session() as sess:
  reader = pywrap_tensorflow.NewCheckpointReader(tf_checkpoint_filename)
  if all_tensors:
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in sorted(var_to_shape_map):

      if ('biases' in key) and not('Momentum' in key):

        caffe_layername=key.replace('/biases','') 
        print("tensor_name: ", key)
        print("caffelayer_name: ",caffe_layername)
        print("\n")
        bias_tf=reader.get_tensor(key)
        net.params[caffe_layername][1].data[:]=bias_tf

      elif ('kernels' in key) and not('Momentum' in key):

        caffe_layername=key.replace('/kernels','') 
        print("tensor_name: ", key)
        print("caffelayer_name: ",caffe_layername)
        print("\n")
        kernel_tf=reader.get_tensor(key)
        kernel_caffe=np.transpose(kernel_tf,[3,2,0,1])  
        net.params[caffe_layername][0].data[:]=kernel_caffe

net.save('SqueezeDet.caffemodel')
#for k, v in net.params.items():
#	print(k, v[0].data, v[1].data)

    
