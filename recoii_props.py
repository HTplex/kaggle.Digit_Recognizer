def fp_full_bias():

def bp_full_bias():


def one_layer_training_with_bias_api(alpha, train_batch, train_label_batch, in_to_hid_weights, hid_bias,
                                     hid_to_out_weights, out_bias, batch_size):
    # fprop
    hid_state = np.dot(train_batch, in_to_hid_weights)
    hid_state = hid_state + hid_bias
    out_state = np.dot(hid_state, hid_to_out_weights)
    out_state = out_state + out_bias
    out_state = 1 / (1 + np.exp(-out_state))

    # bprop:err
    err = out_state - train_label_batch
    CE = (err * err / 2).sum(axis=1)
    d_Out = out_state * (1 - out_state) * err
    d_hid_to_out = np.dot(hid_state.transpose(), d_Out)
    d_hid = np.dot(d_Out, hid_to_out_weights.transpose())
    d_in_to_hid = np.dot(train_batch.transpose(), d_hid)
    d_in = np.dot(d_hid, in_to_hid_weights.transpose())

    # update weights
    del_hid_to_out = -1 * alpha * d_hid_to_out
    del_in_to_hid = -1 * alpha * d_in_to_hid
    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid

    hid_bias += -1 * alpha * d_hid
    out_bias += -1 * alpha * d_Out

    return in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias, d_in
