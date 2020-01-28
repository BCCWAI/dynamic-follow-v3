import tensorflow as tf
from keras import backend as K
from tensorflow.python.platform import gfile

f = gfile.FastGFile("C:/Git/dynamic-follow-tf/models/pb_models/15-min-5.pb", 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()
sess=K.get_session()
sess.graph.as_default()
tf.import_graph_def(graph_def)

print([n.name for n in tf.get_default_graph().as_graph_def().node])

'''softmax_tensor = sess.graph.get_tensor_by_name('import/dense_9/BiasAdd:0')

predictions = sess.run(softmax_tensor, {'import/dense_1_input:0': [[1,.5,.32,.1,.9]]})
print([n.name for n in tf.get_default_graph().as_graph_def().node])
print(predictions)'''