def get_last_inception_output_size(inception_out, inception_depth):
    return inception_out * 4 * (2 ** (inception_depth - 1))