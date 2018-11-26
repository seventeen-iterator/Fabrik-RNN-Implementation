import tensorflow as tf
from tensorflow.contrib.util import constant_value
from google.protobuf import text_format
import sys

try:
    model_file = sys.argv[1]
except IndexError:
    print("Syntax: python explore_tensorflow_model.py <path/to/pbtxt>")

with open(model_file, 'r') as model_file:
    model_protobuf = text_format.Parse(model_file.read(),
                                        tf.GraphDef())

tf.import_graph_def(model_protobuf)



graph = tf.get_default_graph()
tensors = [n for n in graph.as_graph_def().node]
for n in tensors:
    if n.op == "Const":
        t = graph.get_tensor_by_name(n.name + ':0')
        print(n.name, constant_value(t))
