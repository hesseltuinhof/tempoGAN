import tensorflow as tf
import numpy as np
import argparse
import os
from tempogan import Network
from config import config


def main(testdata, outputfile):

    # define model
    model = Network(testdata, config)

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # start evaluation
    inputs, truth, outputs = model.predict(sess, testdata)

    output_path = os.path.join(outputfile)
    np.savez_compressed(output_path,
                        outputs=np.squeeze(outputs),
                        inputs=np.squeeze(inputs),
                        truths=np.squeeze(truth))
    print("[*] Saved predictions of testdata : {} to {}"
          .format(testdata, outputfile))

    print("Finished")
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',
                        default='../data/predictions.npz',
                        help='Name of output file to save predictions.')
    parser.add_argument('--input',
                        default='../data/test.tfrecords',
                        help='Name of tfrecord file containing testdata.')
    args = vars(parser.parse_args())
    main(args['input'], args['out'])
