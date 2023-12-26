import numpy as np
import config


def isDebug():
    import sys
    return True if sys.gettrace() else False


def suppress_warning():
    import logging
    import os
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def platform():
    import platform
    return platform.system()


def reduce_gpu_memory_usage():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def get_current_pid():
    import os
    return os.getpid()