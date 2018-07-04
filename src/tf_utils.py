import tensorflow as tf


def convolution3d(inputs, filters, kernel_size, strides=(1, 1, 1),
                  padding="same", name="conv3d", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        outputs = tf.layers.conv3d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
            reuse=reuse)
    return outputs


def residual_block(inputs, filters, name="residual", norm=None, reuse=False):
    """
    Implementation of the residual block.
    Reference: Deep Residual Learning for Image Recognition (He et al. 2015)
    """
    assert filters[1] == filters[2]

    with tf.variable_scope(name, reuse=reuse):
        left = convolution3d(inputs, filters[0], 5, name=name+"_convA",
                             reuse=reuse)
        if norm is not None:
            left = normalization(left, name=name+"_norm1", type=norm,
                                 reuse=reuse)
        left = tf.nn.relu(left, name=name+"_relu1")
        left = convolution3d(left, filters[1], 5, name=name+"_convB",
                             reuse=reuse)
        if norm is not None:
            left = normalization(left, name=name+"_norm2", type=norm,
                                 reuse=reuse)

        right = convolution3d(inputs, filters[2], 1, name=name+"_convS",
                              reuse=reuse)
        if norm is not None:
            right = normalization(right, name=name+"_norm3", type=norm,
                                  reuse=reuse)

        outputs = tf.nn.relu(left + right, name=name+"_relu2")
    return outputs


def downsample(inputs):
    outputs = tf.nn.max_pool(inputs, ksize=2, strides=2,
                             padding="SAME", name="down")
    return outputs


def normalization(inputs, name="norm", reuse=False, type="batch",
                  training=True):
    """
    Implementation of instance normalization, which is used instead of batch
    normalization.
    Reference: Instance normalization: The Missing Ingredient for Fast
               Stylization (Ulyanov et. al 2017)
    """
    if type == "instance":
        with tf.variable_scope(name, reuse=reuse):
            depth = inputs.get_shape()[3]
            gamma = tf.get_variable(
                    "scale",
                    [depth],
                    initializer=tf.random_normal_initializer(1.0, 0.01))
            beta = tf.get_variable(
                    "offset",
                    [depth],
                    initializer=tf.constant_initializer(1.0))
            mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
            std_inverse = tf.rsqrt(variance + 0.00001)
            outputs = gamma * (inputs - mean) * std_inverse + beta
    else:
        with tf.variable_scope(name, reuse=reuse):
            outputs = tf.layers.batch_normalization(inputs,
                                                    training=training,
                                                    reuse=reuse)
    return outputs


def upsample3d(inputs, factor=2, name="upsample"):
    outputs = tf.keras.layers.UpSampling3D(
                size=(factor, factor, factor), name=name)(inputs)
    return outputs
