from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.util import constant_value
from init_detector import detect_init
from google.protobuf import text_format
import sys
import re


def fetch_reg(reg_tensors, tensor_to_reg):
    """This function determines whether in array exists regularizer for given tensor"""
    return len([n for n in reg_tensors if n.input[0] == tensor_to_reg])


def get_layer_params(layer_name, sess, graph, alpha, epsilon):
    REG_MAP = {
        (0, 0): 'None',
        (1, 0): 'l1',
        (0, 1): 'l2',
        (1, 1): 'l1l2'
    }

    tensors = graph.as_graph_def().node
    layer_re = re.compile(r'import/' + layer_name + '_(\d)+/.*')

    # possible variables of recurrent layer are kernel, recurrent kernel and bias
    # after running init operator there will be some value for analysis
    rec_tensors = [n.name
                   for n in tensors
                   if layer_re.search(n.name) and n.op == "VariableV2"]
    n_layers = len(rec_tensors) // 3
    
    # l1 regularization uses abs operation, while l2 uses square operation
    # In lists below there are operations that might be used by regularization
    l1 = [n for n in tensors if layer_re.search(n.name) and n.op == "Abs"]
    l2 = [n for n in tensors if layer_re.search(n.name) and n.op == "Square"]
    
    print("Detected", layer_name, "layers:")
    for i in range(n_layers):
        kernel_name, rkernel_name, bias_name = rec_tensors[i*3:i*3+3]
        layer_i = layer_name + '_' + layer_re.search(kernel_name).group(1)
        layer_i_re = re.compile(r'import/' + layer_i + '/.*')

        k_var = graph.get_tensor_by_name(kernel_name + ':0')
        rk_var = graph.get_tensor_by_name(rkernel_name + ':0')
        b_var = graph.get_tensor_by_name(bias_name + ':0')
        kernel = sess.run(k_var)
        rkernel = sess.run(rk_var)
        bias = sess.run(b_var)

        outputs = rkernel.shape[0]
        kernel_init = detect_init(kernel, alpha, epsilon)
        rkernel_init = detect_init(rkernel, alpha, epsilon)
        bias_init = detect_init(bias, alpha, epsilon)
        
        l1_i = [n for n in l1 if layer_i_re.search(n.name)]
        l2_i = [n for n in l2 if layer_i_re.search(n.name)]
        k_read = 'import/' + layer_i + '/kernel/read'
        rk_read = 'import/' + layer_i + '/recurrent_kernel/read'
        k_reg = REG_MAP[(fetch_reg(l1_i, k_read), fetch_reg(l2_i, k_read))]
        rk_reg = REG_MAP[(fetch_reg(l1_i, rk_read), fetch_reg(l2_i, rk_read))]
        
        prob_name = 'import/' + layer_i + '/dropout/keep_prob:0'
        rprob_name = 'import/' + layer_i + '/cond/dropout/keep_prob:0'
        
        # Dropout is 1 - keep probability
        # keep probability tensor extracted by name
        # But if there's no tensor - it means that there's no dropout
        # Extraction made via constant_value from tf.contrib.util instead of sess.run
        # to bypass necessity of feeding values
        try:
            prob_tensor = graph.get_tensor_by_name(prob_name)
            dropout = 1 - float(constant_value(prob_tensor))
        except KeyError:
            dropout = 0.0

        try:
            rprob_tensor = graph.get_tensor_by_name(rprob_name)
            rdropout = 1 - float(constant_value(rprob_tensor))
        except KeyError:
            rdropout = 0.0

        print('- Layer', layer_i + ':')
        print('-- Outputs:', outputs)
        print('-- Kernel init:', kernel_init)
        print('-- Recurrent kernel init:', rkernel_init)
        print('-- Bias init:', bias_init)
        print('-- Kernel regularization:', k_reg)
        print('-- Recurrent kerner regularization:', rk_reg)
        print('-- Dropout:', dropout)
        print('-- Recurrent dropout:', rdropout)


try:
    model_file_name = sys.argv[1]
except IndexError:
    print('Usage: python rnn_detector.py <MODEL_FILE.pbtxt>')

with open(model_file_name, 'r') as model_file:
    model_protobuf = text_format.Parse(model_file.read(),
                                       tf.GraphDef())

tf.import_graph_def(model_protobuf)
graph = tf.get_default_graph()

tensors = graph.as_graph_def().node
init = [n.name for n in tensors if n.op == "NoOp"][0]
init_tensor = graph.get_operation_by_name(init)

ALPHA = 0.05
EPSILON = 0.01

with tf.Session(graph=graph) as sess:
    sess.run(init_tensor)
    
    args = (sess, graph, ALPHA, EPSILON)
    get_layer_params('simple_rnn', *args)
    get_layer_params('gru', *args)
    get_layer_params('lstm', *args)

