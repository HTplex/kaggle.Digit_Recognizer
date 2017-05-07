import numpy as np
def fprop_1lr(input_layer_state, input_to_hid_weights, hid_layer_state, hid_to_out_weights, hid_bias, out_bias):
    hid_layer_state = np.dot(input_layer_state, input_to_hid_weights)
    hid_layer_state = hid_layer_state + hid_bias
    out_layer_state = np.dot(hid_layer_state, hid_to_out_weights)
    out_layer_state = out_layer_state + out_bias
    return [hid_layer_state, out_layer_state]


def bprop_1lr(train_label, out_layer_state, hid_to_out_weights, hid_layer_state, input_to_hid_weights, hid_bias,
              out_bias, learn_rate):
    err_out = -(train_label-out_layer_state)
    err_dev_out = err_out*(1-err_out)
    err_dev_hid_to_out = err_dev_out