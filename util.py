import numpy
import tensorflow as tf

def seed_random_number_generators():
    """ seed random number generator for tensorflow
    """
    from numpy.random import seed
    seed(1)
    tf.random.set_seed(2)
    pass
