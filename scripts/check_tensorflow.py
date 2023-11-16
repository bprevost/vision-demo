#!/usr/bin/env python3

import tensorflow as tf

if tf.test.is_built_with_cuda():
    print('Built with CUDA!')
else:
    print('Not built with CUDA!')

print(tf.config.list_physical_devices('GPU'))
