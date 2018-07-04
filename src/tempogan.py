import tensorflow as tf
import numpy as np
import os
import sys
from records import get_data

from tf_utils import upsample3d
from tf_utils import convolution3d
from tf_utils import residual_block
from tf_utils import downsample
from tf_utils import normalization


class Network():
    def __init__(self, datafile, config, debug=False):
        self.num_epochs = config.num_epochs
        self.debug = debug
        self.ckpt_dir = config.ckpt_dir

        data = get_data(datafile, config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(data.output_types,
                                                        data.output_shapes)
        self.train_init_op = self.iterator.make_initializer(data)

        # densities, velocities
        dtc, dtp, dtn, Dtc, Dtp, Dtn, vp, vn = self.iterator.get_next()
        self.dtc, self.Dtc = dtc, Dtc

        # generator
        G = generator(dtc, name="G", reuse=False)
        self.G = G
        Gp = generator(dtp, name="G", reuse=True)
        Gn = generator(dtn, name="G", reuse=True)

        # spatial discriminator
        D_s = discriminator_spatial(dtc, Dtc, name="D_s", reuse=False)
        D_s_g = discriminator_spatial(dtc, G, name="D_s", reuse=True)

        # temporal discriminator
        D_t = discriminator_temporal([Dtp, Dtc, Dtn],
                                     [vp, vn],
                                     name="D_t",
                                     reuse=False)
        D_t_g = discriminator_temporal([Gp, G, Gn],
                                       [vp, vn],
                                       name="D_t",
                                       reuse=True)
        # losses
        self.loss_D_s = -tf.reduce_mean(tf.log(D_s)) \
            - tf.reduce_mean(tf.log(tf.ones_like(D_s_g)
                                    - D_s_g*config.label_smooth))

        self.loss_D_t = -tf.reduce_mean(tf.log(D_t)) \
            - tf.reduce_mean(tf.log(tf.ones_like(D_t_g)
                                    - D_t_g*config.label_smooth))

        self.loss_G = -tf.reduce_mean(tf.log(D_s_g) + tf.log(D_t_g)) \
            + config.lambda_L * tf.reduce_mean(tf.norm(G - Dtc))

        # get variable names
        trainable_vars = tf.trainable_variables()
        G_var = [var for var in trainable_vars if "G" in var.name]
        D_s_var = [var for var in trainable_vars if "D_s" in var.name]
        D_t_var = [var for var in trainable_vars if "D_t" in var.name]

        # optimizer
        self.G_optim = tf.train.AdamOptimizer(config.learning_rate,
                                              beta1=config.beta1)
        self.G_optim = self.G_optim.minimize(self.loss_G, var_list=G_var)

        self.D_s_optim = tf.train.AdamOptimizer(config.learning_rate,
                                                beta1=config.beta1)
        self.D_s_optim = self.D_s_optim.minimize(self.loss_D_s,
                                                 var_list=D_s_var)

        self.D_t_optim = tf.train.AdamOptimizer(config.learning_rate,
                                                beta1=config.beta1)
        self.D_t_optim = self.D_t_optim.minimize(self.loss_D_t,
                                                 var_list=D_t_var)

        # bounds
        self.bounds = [tf.reduce_min(D_s), tf.reduce_max(D_s),
                       tf.reduce_min(D_s_g), tf.reduce_max(D_s_g),
                       tf.reduce_min(D_t), tf.reduce_max(D_t),
                       tf.reduce_min(D_t_g), tf.reduce_max(D_t_g)]

        # set loss tensorboard summary
        self.tb_loss_G = tf.placeholder(tf.float32, [], name="loss_G")
        self.tb_loss_D_s = tf.placeholder(tf.float32, [], name="loss_D_s")
        self.tb_loss_D_t = tf.placeholder(tf.float32, [], name="loss_D_t")
        tf.summary.scalar("loss_G", self.tb_loss_G)
        tf.summary.scalar("loss_D_s", self.tb_loss_D_s)
        tf.summary.scalar("loss_D_t", self.tb_loss_D_t)
        self.summary = tf.summary.merge_all()
        log_path = os.path.join(self.ckpt_dir, 'tensorboard')
        self.writer = tf.summary.FileWriter(log_path,
                                            graph=tf.get_default_graph())

    def fit(self, sess, save_interval=50):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("[!] Using variables from %s" % ckpt.model_checkpoint_path)
        else:
            print("[!] Initialized variables")

        for epoch in range(self.num_epochs):
            Loss_G, Loss_D_s, Loss_D_t = 0, 0, 0
            step = 0
            sess.run(self.train_init_op)
            while True:
                try:
                    _, _, _, _loss_G, _loss_D_s, _loss_D_t, debug_vals = sess.run(
                            [self.G_optim,
                             self.D_s_optim,
                             self.D_t_optim,
                             self.loss_G,
                             self.loss_D_s,
                             self.loss_D_t,
                             self.bounds])

                    if step % 5 == 0 and self.debug:
                        print("loss_G: {:.4f}, loss_D_s: {:.4f}, loss_D_t: "
                              + "{:.4f}".format(_loss_G, _loss_D_s, _loss_D_t))
                        print("D_s: [{:.4f}, {:.4f}] D_s_g: [{:.4f}, {:.4f}]\n"
                              + "D_t: [{:.4f}, {:.4f}] D_t_g: [{:.4f}, {:.4f}]"
                              .format(*debug_vals))

                    Loss_G += _loss_G
                    Loss_D_s += _loss_D_s
                    Loss_D_t += _loss_D_t
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

                except KeyboardInterrupt:
                    ckpt_path = os.path.join(self.ckpt_dir, 'ckpt')
                    save_path = saver.save(sess, ckpt_path, global_step=epoch)
                    print("Interrupt, model saved to: ", save_path)
                    sys.exit()

            Loss_G /= step
            Loss_D_s /= step
            Loss_D_t /= step
            print("[*] epoch {}/{} loss_G: {:.4f}  loss_D_s: {:.4f}  loss_D_t:"
                  + " {:.4f}".format(epoch+1, self.num_epochs, Loss_G,
                                     Loss_D_s, Loss_D_t))

            summary = sess.run(
                    self.summary,
                    feed_dict={self.tb_loss_G: Loss_G,
                               self.tb_loss_D_s: Loss_D_s,
                               self.tb_loss_D_t: Loss_D_t})
            self.writer.add_summary(summary)
            if epoch % save_interval or epoch == self.num_epochs:
                ckpt_path = os.path.join(self.ckpt_dir, 'ckpt')
                save_path = saver.save(sess, ckpt_path, global_step=epoch)
                print("[*] Model saved in %s" % save_path)

    def predict(self, sess, datafile):
        data = get_data(datafile, 1)
        init_op = self.iterator.make_initializer(data)
        outputs, inputs, truth = [], [], []
        sess.run(init_op)
        while True:
            try:
                G, Dtc, dtc = sess.run([self.G, self.Dtc, self.dtc])
                outputs.append(np.array(G))
                inputs.append(np.array(dtc))
                truth.append(np.array(Dtc))
            except tf.errors.OutOfRangeError:
                break
        outputs = np.stack(outputs, axis=0)
        inputs = np.stack(inputs, axis=0)
        truth = np.stack(truth, axis=0)
        return inputs, truth, outputs


def generator(inputs, name="generator", reuse=False):
    """
    Implementation of the generator architecture. Namely:
    in, NI, RB_0, RB_1, RB_2, RB_3, out
    Reference: arXiv:1801.09710
    """
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            tf.variable_scope(scope, reuse=False)
            assert scope.reuse is False, "'scope.reuse' should be False"

    # upsampling part
    up1 = upsample3d(inputs, 4, name+"_NI")

    # residual part
    res1 = residual_block(up1, [8, 32, 32], name=name+"_RB_0", reuse=reuse)
    res2 = residual_block(res1, [128, 128, 128], name=name+"_RB_1",
                          reuse=reuse)
    res3 = residual_block(res2, [32, 8, 8], name=name+"_RB_2", reuse=reuse)
    res4 = residual_block(res3, [2, 1, 1], name=name+"_RB_3", reuse=reuse)

    return res4


def discriminator_spatial(x, y, name="discriminator_spatial", norm="batch",
                          reuse=False):
    """
    Implementation of the spatial discriminator architecture. Namely:
    in_x, NI|in_y, C1, C2, C3, C4, FC, out
    Reference: arXiv:1801.09710
    """
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            tf.variable_scope(scope, reuse=False)
            assert scope.reuse is False, "'scope.reuse' should be False"

    # input part
    up = upsample3d(x, 4, name+"_NI")
    concat = tf.concat([up, y], axis=-1, name=name+"_concat")

    # convolution part
    conv1 = convolution3d(concat, 32, 4, strides=(2, 2, 2),
                          name=name+"_conv1", reuse=reuse)
    lrelu1 = tf.nn.leaky_relu(conv1, name=name+"_lrelu1")

    conv2 = convolution3d(lrelu1, 64, 4, strides=(2, 2, 2),
                          name=name+"_conv2", reuse=reuse)
    norm2 = normalization(conv2, name=name+"_norm2", type=norm, reuse=reuse)
    lrelu2 = tf.nn.leaky_relu(norm2, name=name+"_lrelu2")

    conv3 = convolution3d(lrelu2, 128, 4, strides=(2, 2, 2),
                          name=name+"_conv3", reuse=reuse)
    norm3 = normalization(conv3, name=name+"_norm3", type=norm, reuse=reuse)
    lrelu3 = tf.nn.leaky_relu(norm3, name=name+"_lrelu3")

    conv4 = convolution3d(lrelu3, 256, 4, name=name+"_conv4", reuse=reuse)
    norm4 = normalization(conv4, name=name+"_norm4", type=norm, reuse=reuse)
    lrelu4 = tf.nn.leaky_relu(norm4, name=name+"_lrelu4")

    # output part
    fc5 = tf.layers.dense(tf.layers.flatten(lrelu4), units=1,
                          name=name+"_fc5", reuse=reuse)
    output = tf.sigmoid(fc5, name=name+"out")

    return output


def advection(y, v, name="advection"):
    # TODO: implement
    return y


def discriminator_temporal(Y, V, name="discriminator_temporal", norm="batch",
                           reuse=False):
    """
    Implementation of the temporal discriminator architecture. Namely:
    in_y, C1, C2, C3, C4, FC, out
    Reference: arXiv:1801.09710
    """

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            tf.variable_scope(scope, reuse=False)
            assert scope.reuse is False, "'scope.reuse' should be False"

    # advection part
    adv1 = advection(Y[0], V[0], name=name+"_advection_forward")
    adv2 = advection(Y[2], V[1], name=name+"_advection_backward")

    conv_inputs = tf.concat([adv1, Y[1], adv2], axis=-1, name=name+"_concat")

    # convolution part
    conv1 = convolution3d(conv_inputs, 32, 4, strides=(2, 2, 2),
                          name=name+"_conv1", reuse=reuse)
    lrelu1 = tf.nn.leaky_relu(conv1, name=name+"_lrelu1")

    conv2 = convolution3d(lrelu1, 64, 4, strides=(2, 2, 2),
                          name=name+"_conv2", reuse=reuse)
    norm2 = normalization(conv2, name=name+"_norm2", type=norm, reuse=reuse)
    lrelu2 = tf.nn.leaky_relu(norm2, name=name+"_lrelu2")

    conv3 = convolution3d(lrelu2, 128, 4, strides=(2, 2, 2),
                          name=name+"_conv3", reuse=reuse)
    norm3 = normalization(conv3, name=name+"_norm3", type=norm, reuse=reuse)
    lrelu3 = tf.nn.leaky_relu(norm3, name=name+"_lrelu3")

    conv4 = convolution3d(lrelu3, 256, 4, name=name+"_conv4", reuse=reuse)
    norm4 = normalization(conv4, name=name+"_norm4", type=norm, reuse=reuse)
    lrelu4 = tf.nn.leaky_relu(norm4, name=name+"_lrelu4")

    # output part
    fc5 = tf.layers.dense(tf.layers.flatten(lrelu4), units=1, activation=None,
                          name=name+"_fc5", reuse=reuse)
    output = tf.sigmoid(fc5, name=name+"out")

    return output
