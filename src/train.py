import tensorflow as tf
import numpy as np
import argparse
import os
from tempogan import Network
from config import config


def main(datafile, testdata):
    # define model
    model = Network(datafile, config)

    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model.fit(sess, config.save_interval)

    if testdata:
        inputs, truth, outputs = model.predict(sess, testdata)

        ckpt_path = os.path.join(config.ckpt_dir, 'predictions.npz')
        np.savez_compressed(ckpt_path,
                            outputs=np.squeeze(outputs),
                            inputs=np.squeeze(inputs),
                            truths=np.squeeze(truth))
        print("[*] Saved predictions of testdata : {} to {}"
              .format(testdata, ckpt_path))

    print("Finished")
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default='../data/train.tfrecords',
                        help='Name of tfrecord file containing training data.')
    parser.add_argument('--testdata',
                        default=None,
                        help='Name of tfrecord file containing validation data.')
    args = vars(parser.parse_args())
    main(args['input'], args['testdata'])
