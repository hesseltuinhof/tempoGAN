# adapted from
# https://bitbucket.org/mantaflow/manta/src/15eaf4aa72da62e174df6c01f85ccd66fde20acc/tensorflow/tools/uniio.py

import gzip
import struct
import numpy as np

from collections import namedtuple


def read_uni(filename):
    """
    Read a '*.uni' file. Returns the header as a dictionary and the content as
    a numpy-array.
    """
    with gzip.open(filename, 'rb') as bytestream:
        header = _read_uni_header(bytestream)
        array = _read_uni_array(bytestream, header)
    return header, array


def _read_uni_header(bytestream):
    header_type = bytestream.read(4).decode("utf-8")
    assert header_type == "MNT3", "Header not supported"
    HeaderV4 = namedtuple(
        'HeaderV4',
        'dimX dimY dimZ gridType elementType bytesPerElement info '
        + 'dimT timestamp')
    header = HeaderV4._asdict(
        HeaderV4._make(struct.unpack('iiiiii252siQ', bytestream.read(288))))
    return header


def _read_uni_array(bytesstream, header):
    assert (header['bytesPerElement'] == 12 and header['elementType'] == 2) \
        or (header['bytesPerElement'] == 4 and (header['elementType'] == 0
            or header['elementType'] == 1)), "Header mismatch"

    if header['elementType'] == 0:
        data = np.frombuffer(bytesstream.read(), dtype='int32')
    else:
        data = np.frombuffer(bytesstream.read(), dtype='float32')

    if header['elementType'] == 2:
        channels = 3
    else:
        channels = 1

    if header['dimT'] <= 1:
        dimensions = [header['dimZ'], header['dimY'], header['dimX'], channels]
    else:
        dimensions = [header['dimT'],
                      header['dimZ'],
                      header['dimY'],
                      header['dimX'],
                      channels]

    return data.reshape(*dimensions, order='C')
