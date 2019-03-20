import os
import pickle
import copy
import numpy as np
import tensorflow as tf
from collections import Counter
import preprocess as pre

def main():
    
    (source_text, target_text), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab), (source_data, target_data) = load_data('preprocess.p')

    save_path = 'H:\gsq-metadata-extraction\Seq2Seq Example\checkpoints\dev'
    checkpoint_file=tf.train.latest_checkpoint(save_path)
    graph=tf.Graph()

    # get text and convert
    # source_text = input("English: ")
    source_text = "my favorite animal is my cat ."
    source_ints = pre.text_to_ints(source_text, source_vocab_to_int)

    with graph.as_default():
        session_conf = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess,checkpoint_file)
            inputs = graph.get_operation_by_name("input").outputs[0]
            prediction = graph.get_operation_by_name("prediction").outputs[0]
            

            print(sess.run(prediction, feed_dict={input:source_ints}))


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__": main()