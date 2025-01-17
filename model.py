from __future__ import division

import itertools
import os
import time
from glob import glob


from six.moves import xrange

from ops import *
from utils import *

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]


def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64, lowres=8,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim

        self.d_bns = [batch_norm(name='d_bn{}'.format(i, )) for i in range(4)]

        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"


    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.lowres_images = tf.reduce_mean(tf.reshape(self.images, [self.batch_size, self.lowres_size, self.lowres, self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        self.lowres_G = tf.reduce_mean(tf.reshape(self.G, [self.batch_size, self.lowres_size, self.lowres, self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.lowres_mask = tf.placeholder(tf.float32, self.lowres_shape, name='lowres_mask')
        self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        self.contextual_loss += tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))),1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        data = dataset_files(config.dataset)
        """"
        data = list()

        for i in tmp_data:
            image = get_image(i, self.image_size, is_crop=self.is_crop)
            #       #image = scipy.misc.imread(i, mode='RGB').astype(np.float)
            for x in xrange(0, image.shape[0], 64):
                for y in xrange(0, image.shape[1], 64):
                    mxX = min(x + 64, image.shape[0])
                    mxY = min(y + 64, image.shape[1])
                    isGood = True
                    for tx in xrange(mxX - 64, mxX):
                        for ty in xrange(mxY - 64, mxY):
                            def almost_equal(a, b, epsilon=0.7):
                                return abs(a - b) <= epsilon

                            if almost_equal(image[tx, ty, 0], 1) and almost_equal(image[tx, ty, 1], -1) and almost_equal(image[tx, ty, 2], 1):
                                isGood = False
                    if isGood:
                        imgName = 'train/tmp{:s}.png'.format(str(i.__hash__()) + str(x) + str(y))
                        save_images(np.array([image[mxX - 64: mxX, mxY - 64: mxY, :]]).astype(np.float32), [1, 1], imgName)
                        data.append(imgName)
        """
        np.random.shuffle(data)
        assert (len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("model in the checkpoint")
        else:
            print("new model")

        for epoch in xrange(config.epoch):
           # data = dataset_files(config.dataset)

            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images: batch_images, self.z: batch_z,
                                                          self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )
                    save_images(samples, [8, 8], './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def complete(self, config):
        def make_dir(name):
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)

        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert isLoaded

        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs / self.batch_size))
        lowres_mask = np.zeros(self.lowres_shape)

        for idx in xrange(0, batch_idxs):
            l = idx * self.batch_size
            u = min((idx + 1) * self.batch_size, nImgs)
            batchSz = u - l
            batch_files = config.imgs[l:u]

            batch0 = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]

            a = 0
            b = 0
            queue1 = list()
            queue2 = list()
            queue3 = list()
            queue4 = list()
            step = 50
            n = batch0[0].shape[0]
            m = batch0[0].shape[1]

            while a <= m and b <= m:
                def func(x, y, q):
                    mxX = min(x + 64, batch0[0].shape[0])
                    mxY = min(y + 64, batch0[0].shape[1])
                    if len(q) == 0 or q[-1] != (mxX, mxY):
                        q.append((mxX, mxY))
                    return mxX, mxY

                for k in xrange(a, n, step):
                    func(k, b, queue1)
                for k in xrange(b + step, m, step):
                    func(n, k, queue2)
                for k in xrange(n - step, a, -step):
                    func(k, n, queue3)
                for k in xrange(m - step, b + step, -step):
                    func(a, k, queue4)

                a += step
                b += step
                n -= step
                m -= step

            queue1.reverse()
            queue2.reverse()
            queue3.reverse()
            queue4.reverse()

            for q in [queue1, queue2, queue3, queue4]:
                for cnt in q:
                    maxX, maxY = cnt

                    batch = [(batch0[0][maxX - 64: maxX, maxY - 64:maxY, :])]
                    isNeededToReconstuct = False

                    mask = np.ones(self.image_shape)
                    for i in range(0, self.image_size):
                        for j in range(0, self.image_size):
                            def almost_equal(x, y, epsilon=0.7):
                                return abs(x - y) <= epsilon
                            if almost_equal(batch[0][i][j][0], 1.0) and almost_equal(batch[0][i][j][1], -1.0) and almost_equal(batch[0][i][j][2], 1.0):
                                mask[i, j, :] = 0.0
                                isNeededToReconstuct = True

                    if not isNeededToReconstuct:
                        continue

                    batch_images = np.array(batch).astype(np.float32)
                    if batchSz < self.batch_size:
                        print(batchSz)
                        padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
                        batch_images = np.pad(batch_images, padSz, 'constant')
                        batch_images = batch_images.astype(np.float32)

                    zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                    m = 0
                    v = 0

                    nRows = np.ceil(batchSz / 8)
                    nCols = min(8, batchSz)
                    save_images(batch_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'before.png'))
                    masked_images = np.multiply(batch_images, mask)
                    # masked_images = batch_images
                    save_images(masked_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'masked.png'))

                    if lowres_mask.any():
                        lowres_images = np.reshape(batch_images,
                                                   [self.batch_size, self.lowres_size, self.lowres, self.lowres_size,
                                                    self.lowres, self.c_dim]).mean(4).mean(2)
                        lowres_images = np.multiply(lowres_images, lowres_mask)
                        lowres_images = np.repeat(np.repeat(lowres_images, self.lowres, 1), self.lowres, 2)
                        save_images(lowres_images[:batchSz, :, :, :], [nRows, nCols], os.path.join(config.outDir, 'lowres.png'))

                    for img in range(batchSz):
                        with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                            f.write('iter loss ' + ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) + '\n')

                    loss = 0
                    for i in xrange(config.nIter):
                        fd = {
                            self.z: zhats,
                            self.mask: mask,
                            self.lowres_mask: lowres_mask,
                            self.images: batch_images,
                            self.is_training: False
                        }
                        run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                        loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)

                        for img in range(batchSz):
                            with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                                f.write('{} {} '.format(i, loss[img]).encode())
                                np.savetxt(f, zhats[img:img + 1])

                        if i % config.outInterval == 0:
                            print(i, np.mean(loss[0:batchSz]))
                            inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
                            completed = masked_images + inv_masked_hat_images

                            for i0 in range(maxX - 64, maxX):
                                for j0 in range(maxY - 64, maxY):
                                    batch0[0][i0][j0] = completed[0][i0 - maxX + 64][j0 - maxY + 64]

                            imgName = os.path.join(config.outDir, 'completed/{:04d}.png'.format(i))
                            save_images(completed[:batchSz, :, :, :], [nRows, nCols], imgName)

                        # Optimize single completion with Adam
                        m_prev = np.copy(m)
                        v_prev = np.copy(v)
                        m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                        v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                        m_hat = m / (1 - config.beta1 ** (i + 1))
                        v_hat = v / (1 - config.beta2 ** (i + 1))
                        zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                        zhats = np.clip(zhats, -1, 1)

                    if loss > 700:
                        break

            imgName = os.path.join(config.outDir, 'completed/finale.png')
            save_images(np.array(batch0).astype(np.float32), [1, 1], imgName)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=True)

            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1
            depth_mul = 8
            size = 8

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i - 1], [self.batch_size, size, size, self.gf_dim * depth_mul], name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1], [self.batch_size, size, size, 3], name=name, with_w=True)

            return tf.nn.tanh(hs[i])

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
