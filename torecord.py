import tensorflow as tf

from glob import glob
from utils import read_uni


def write_uni_to_record(folderprefix, filename='data.tfrecords'):
    """
    Convert a folder containing 'density_*.uni' and 'vel_*.uni' files
    into a TFRecords file.
    """
    assert filename.endswith('.tfrecords'), "Wrong file extension"
    writer = tf.python_io.TFRecordWriter(filename)

    densities, velocities = _get_uni_list(folderprefix)
    assert len(densities) == len(velocities)

    for i in range(len(densities)):
        density = densities[i]
        velocity = velocities[i]
        N = len(density)
        for j in range(1, N-1):
            dtp, Dtp = _read_file(density[j-1])
            dtc, Dtc = _read_file(density[j])
            dtn, Dtn = _read_file(density[j+1])
            vp, _ = _read_file(velocity[j-1])
            vc, _ = _read_file(velocity[j])
            vn, _ = _read_file(velocity[i+1])
            vp = vc - vp
            vn = vc - vn

            sample = tf.train.Example(features=tf.train.Features(feature={
                'dtp': _bytes_feature(dtp.tostring()),
                'dtc': _bytes_feature(dtc.tostring()),
                'dtn': _bytes_feature(dtn.tostring()),
                'Dtp': _bytes_feature(Dtp.tostring()),
                'Dtc': _bytes_feature(Dtc.tostring()),
                'Dtn': _bytes_feature(Dtn.tostring()),
                'vp': _bytes_feature(vp.tostring()),
                'vn': _bytes_feature(vn.tostring())}))
            writer.write(sample.SerializeToString())

            print("%d/%d" % (j, N), end="\r", flush=True)
        print("\n")
    writer.close()


def _bytes_feature(value):
    bytes_list = tf.train.BytesList(value=[tf.compat.as_bytes(value)])
    return tf.train.Feature(bytes_list=bytes_list)


def _int64_feature(value):
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)


def _get_uni_list(folderprefix=''):
    folders = glob(folderprefix + "_*")
    densities = []
    velocities = []
    for folder in folders:
        densities.append(sorted(glob(folder + "/density_*.uni")))
        velocities.append(sorted(glob(folder + "/vel_*.uni")))
    return (densities, velocities)


def _read_file(f):
    _, x = read_uni(f)
    x_down = x[::4, ::4, ::4, :]
    return x_down, x
