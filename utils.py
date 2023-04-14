import numpy as np
import sympy

from scipy.ndimage import map_coordinates

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


def pearson_mat(matrix, target):
    """
    Assume matrix is of size (samples, x, y)
    and target is of size (samples, 1)
    """

    if len(target.shape) < 3:
        target.shape += (1,) * (3-len(target.shape))
    avg_matrix = matrix.mean(axis=0)
    avg_target = target.mean(axis=0)
    avg_diff_mat = matrix - avg_matrix
    avg_diff_tar = target - avg_target
    term1 = np.sum(avg_diff_mat * avg_diff_tar, axis=0)
    term2 = np.sqrt(np.sum(avg_diff_mat**2, axis=0))
    term3 = np.sqrt(np.sum(avg_diff_tar**2, axis=0))
    term4 = np.fmax(term2*term3, 1e-4*np.ones(term2.shape))
    return term1/term4


def spearman_mat(matrix, target):
    rank_x = np.argsort(matrix, axis=0)
    rank_y = np.argsort(target, axis=0)
    return pearson_mat(rank_x, rank_y)


def range_tuple(matrix):
    return np.nanmin(matrix), np.nanmax(matrix)

def chunks(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i:i+chunk_size]
    
    return None


def polar_to_linear(img, o=None, r=None, output=None, order=1, cont=0, cval=0):
    """
    Taken from https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2
    """    

    if img.ndim == 3:
        if r is None: 
            r = img.shape[0]

        output_original = output

        if output is None:
            output = np.zeros((r*2, r*2, img.shape[2]), dtype=img.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=img.dtype)

        for i in range(img.shape[2]):
            output[:,:,i] = polar_to_linear(img[:,:,i], o, r, output_original, order, cont, cval)
            
    elif img.ndim == 2:
        if r is None: 
            r = img.shape[0]

        if output is None:
            output = np.zeros((r*2, r*2), dtype=img.dtype)
        elif isinstance(output, tuple):
            output = np.zeros(output, dtype=img.dtype)

        if o is None: 
            o = np.array(output.shape)/2 - 0.5

        out_h, out_w = output.shape
        ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
        rs = (ys**2+xs**2)**0.5
        ts = np.arccos(xs/rs)
        ts[ys<0] = np.pi*2 - ts[ys<0]
        ts *= (img.shape[1]-1)/(np.pi*2)

        map_coordinates(img, (rs, ts), order=order, output=output, cval=cval)

        output = np.flip(output, axis=0)

    return output


def linear_to_polar_tf(img_input, radius):
    """
    Not working
    """


    h, w = radius, radius
    cx, cy = img_input.shape[0]//2, img_input.shape[1]//2

    x, y = tf.meshgrid(tf.linspace(0.0, radius, w), tf.linspace(0.0, radius, h))

    x_trans = x - cx
    y_trans = y - cy
    
    r = tf.sqrt(x_trans**2 + y_trans**2)
    theta = tf.atan2(y_trans, x_trans)

    i = tf.round(r * tf.cos(theta) - cx/2)
    # i = np.clip(i, 0, img_input.shape[0]-1)

    j = tf.round(r * tf.sin(theta) - cy/2)
    # j = np.clip(j, 0, img_input.shape[1]-1)

    print(i, j)
    # out_image = n
    out_image = img_input[tf.cast(i, tf.int32), tf.cast(i, tf.int32)]
    return out_image


def square_dims_vector(vector, ratio_w_h=1):
    n = vector.size
    divs = np.array(sympy.divisors(n))
    dist_to_root = np.abs(divs-np.sqrt(n)*ratio_w_h)
    i = np.argmin(dist_to_root)
    x_size = int(divs[i])
    y_size = n//x_size
    return (x_size, y_size) if x_size < y_size else (y_size, x_size)


@keras.utils.register_keras_serializable()
class CylindricalPadding2D(keras.layers.Layer):
    """
    Cylindrical colvolution: https://stackoverflow.com/questions/54911015/keras-convolution-layer-on-images-coming-from-circular-cyclic-domain
    """

    def __init__(self, offset, axis=2, input_dim=32):
        super().__init__()
        self.offset = tf.constant(offset)
        self.axis = tf.constant(axis)
    
    def get_config(self):
        config = super().get_config()
        config["offset"] = int(self.offset)
        return config

    def call(self, inputs):
        extra_right = inputs[:, :, -self.offset:, :]
        extra_left = inputs[:, :, :self.offset, :]
        return tf.concat([extra_right, inputs, extra_left], axis=self.axis)


@keras.utils.register_keras_serializable()
class Masking2D(keras.layers.Layer):
    """
    Cylindrical colvolution: https://stackoverflow.com/questions/54911015/keras-convolution-layer-on-images-coming-from-circular-cyclic-domain
    """

    def __init__(self, mask, def_value=0):
        super().__init__()
        self.mask = tf.constant(mask)
        self.def_value = tf.constant(def_value)
    
    def get_config(self):
        config = super().get_config()
        config["mask"] = self.mask
        config["def_value"] = self.def_value
        return config

    def call(self, input_map):
        return tf.where(self.mask, input_map, self.def_value)