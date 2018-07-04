import tensorflow as tf


def get_data(tfrecordsfile, batch_size=1):
    data = tf.data.TFRecordDataset(tfrecordsfile)

    def parser(record):
        d = [16, 16, 16, 1]
        D = [64, 64, 64, 1]
        features_keys = {'dtp': tf.FixedLenFeature([], tf.string),
                         'dtc': tf.FixedLenFeature([], tf.string),
                         'dtn': tf.FixedLenFeature([], tf.string),
                         'Dtp': tf.FixedLenFeature([], tf.string),
                         'Dtc': tf.FixedLenFeature([], tf.string),
                         'Dtn': tf.FixedLenFeature([], tf.string),
                         'vp': tf.FixedLenFeature([], tf.string),
                         'vn': tf.FixedLenFeature([], tf.string)}
        parsed = tf.parse_single_example(record, features_keys)

        dtp = parse_sample(parsed, 'dtp', d)
        dtc = parse_sample(parsed, 'dtc', d)
        dtn = parse_sample(parsed, 'dtn', d)
        Dtp = parse_sample(parsed, 'Dtp', D)
        Dtc = parse_sample(parsed, 'Dtc', D)
        Dtn = parse_sample(parsed, 'Dtn', D)
        vp = parse_sample(parsed, 'vp', [d[0], d[1], d[2], 3])
        vn = parse_sample(parsed, 'vn', [d[0], d[1], d[2], 3])

        return dtc, dtp, dtn, Dtc, Dtp, Dtn, vp, vn

    data = data.map(parser)
    data = data.batch(batch_size)
    return data


def parse_sample(parsed, key, dims):
    x = tf.decode_raw(parsed[key], tf.float32)
    return tf.reshape(x, dims)
