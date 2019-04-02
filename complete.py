import argparse
import tensorflow as tf

from model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1200)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument('imgs', type=str, nargs='+')

args = parser.parse_args()

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
with tf.device('/gpu:0'):
    with tf.Session(config=config) as sess:
        dcgan = DCGAN(sess, image_size=args.imgSize, batch_size=min(64, len(args.imgs)), checkpoint_dir=args.checkpointDir, lam=args.lam)
        dcgan.complete(args)
