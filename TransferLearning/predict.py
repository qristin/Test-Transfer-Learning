import argparse
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return top_k

def predict(labels, graph, image):
  pil_im = Image.open(image, 'r')
  plt.figure()
  plt.imshow(np.asarray(pil_im))
  
  # load image
  image_data = load_image(image)
  
  # load labels
  labels = load_labels(labels)
  
  # load graph, which is stored in the default session
  load_graph(graph)

  return run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 5)

parser = argparse.ArgumentParser(description='Predict if images belongs to class bowl or vase')

parser.add_argument('--test-image', help='Path to image that needs to be tested')
parser.add_argument('--graph', help='Path to the model graph')
parser.add_argument('--labels', help='Path to labels')

def main():
  args = parser.parse_args()
  predict(args.labels, args.graph, args.test_image)

if __name__ == '__main__':
    main()