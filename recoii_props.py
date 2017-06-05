import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fp_full_linear_bias(input_batch, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias):
    hid_state = hid_state = np.dot(input_batch, in_to_hid_weights)
    hid_state = hid_state + hid_bias
    out_state = np.dot(hid_state, hid_to_out_weights)
    out_state = out_state + out_bias
    return hid_state, out_state


def bp_full_linear_bias(learning_rate, input_batch, in_to_hid_weights, hid_bias, hid_state, hid_to_out_weights,
                        out_bias, d_out):
    d_hid_to_out = np.dot(hid_state.transpose(), d_out)
    d_hid = np.dot(d_out, hid_to_out_weights.transpose())
    d_in_to_hid = np.dot(input_batch.transpose(), d_hid)
    d_in = np.dot(d_hid, in_to_hid_weights.transpose())

    del_hid_to_out = -1 * learning_rate * d_hid_to_out
    del_in_to_hid = -1 * learning_rate * d_in_to_hid
    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid
    hid_bias += -1 * learning_rate * d_hid
    out_bias += -1 * learning_rate * d_out
    return d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias


def bp_full_linear_bias_mmtm(momentum, last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out,  last_del_out_bias,
                             learning_rate, input_batch, in_to_hid_weights, hid_bias, hid_state, hid_to_out_weights,
                             out_bias, d_out):
    d_hid_to_out = np.dot(hid_state.transpose(), d_out)
    d_hid = np.dot(d_out, hid_to_out_weights.transpose())
    d_in_to_hid = np.dot(input_batch.transpose(), d_hid)
    d_in = np.dot(d_hid, in_to_hid_weights.transpose())

    del_hid_to_out = -1 * learning_rate * d_hid_to_out + momentum * last_del_hid_to_out
    del_in_to_hid = -1 * learning_rate * d_in_to_hid + momentum * last_del_in_to_hid
    del_hid_bias = -1 * learning_rate * d_hid + momentum * last_del_hid_bias
    del_out_bias = -1 * learning_rate * d_out + momentum * last_del_out_bias

    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid
    hid_bias += del_hid_bias
    out_bias += del_out_bias

    last_del_hid_to_out = del_hid_to_out
    last_del_in_to_hid = del_in_to_hid
    last_del_hid_bias = del_hid_bias
    last_del_out_bias = del_out_bias
    return last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias,d_in, in_to_hid_weights, hid_bias,\
           hid_to_out_weights, out_bias


def bp_full_linear_bias_mmtm_adplrt(momentum,limit_scale, last_del_in_to_hid, last_d_in_to_hid, lrnrt_in_to_hid, last_del_hid_bias,
                                    last_d_hid, lrnrt_hid_bias, last_del_hid_to_out,  last_d_hid_to_out,
                                    lrnrt_hid_to_out, last_del_out_bias, last_d_out, lrnrt_out_bias,
                                    learning_rate, input_batch, in_to_hid_weights, hid_bias, hid_state,
                                    hid_to_out_weights, out_bias, d_out):
    d_hid_to_out = np.dot(hid_state.transpose(), d_out)
    d_hid = np.dot(d_out, hid_to_out_weights.transpose())
    d_in_to_hid = np.dot(input_batch.transpose(), d_hid)
    d_in = np.dot(d_hid, in_to_hid_weights.transpose())

    for i in range(0,lrnrt_in_to_hid.shape[0]):
        for j in range(0,lrnrt_in_to_hid.shape[1]):
            if (last_d_in_to_hid[i][j]*d_in_to_hid[i][j] > 0) & (lrnrt_in_to_hid[i][j] < limit_scale):
                lrnrt_in_to_hid += 0.05
            if (last_d_in_to_hid[i][j]*d_in_to_hid[i][j] < 0) & (lrnrt_in_to_hid[i][j] > 1/limit_scale):
                lrnrt_in_to_hid *= 0.95

    for i in range(0, lrnrt_hid_to_out.shape[0]):
        for j in range(0, lrnrt_hid_to_out.shape[1]):
            if (last_d_hid_to_out[i][j] * d_hid_to_out[i][j] > 0) & (lrnrt_hid_to_out[i][j] < limit_scale):
                lrnrt_hid_to_out += 0.05
            if (last_d_hid_to_out[i][j] * d_hid_to_out[i][j] < 0) & (lrnrt_hid_to_out[i][j] > 1 / limit_scale):
                lrnrt_hid_to_out *= 0.95

    for i in range(0, lrnrt_hid_bias.shape[0]):
        for j in range(0, lrnrt_hid_bias.shape[1]):
            if (last_d_hid[i][j] * d_hid[i][j] > 0) & (lrnrt_hid_bias[i][j] < limit_scale):
                lrnrt_hid_bias += 0.05
            if (last_d_hid[i][j] * d_hid[i][j] < 0) & (lrnrt_hid_bias[i][j] > 1 / limit_scale):
                lrnrt_hid_bias *= 0.95

    for i in range(0, lrnrt_out_bias.shape[0]):
        for j in range(0, lrnrt_out_bias.shape[1]):
            if (last_d_out[i][j] * d_out[i][j] > 0) & (lrnrt_out_bias[i][j] < limit_scale):
                lrnrt_out_bias += 0.05
            if (last_d_out[i][j] * d_out[i][j] < 0) & (lrnrt_out_bias[i][j] > 1 / limit_scale):
                lrnrt_out_bias *= 0.95

    last_d_hid_to_out = d_hid_to_out
    last_d_hid = d_hid
    last_d_in_to_hid = d_in_to_hid
    last_d_out = d_out

    del_hid_to_out = -1 * learning_rate * lrnrt_hid_to_out * d_hid_to_out + momentum * last_del_hid_to_out
    del_in_to_hid = -1 * learning_rate * lrnrt_in_to_hid * d_in_to_hid + momentum * last_del_in_to_hid
    del_hid_bias = -1 * learning_rate * lrnrt_hid_bias * d_hid + momentum * last_del_hid_bias
    del_out_bias = -1 * learning_rate * lrnrt_out_bias * d_out + momentum * last_del_out_bias

    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid
    hid_bias += del_hid_bias
    out_bias += del_out_bias

    last_del_hid_to_out = del_hid_to_out
    last_del_in_to_hid = del_in_to_hid
    last_del_hid_bias = del_hid_bias
    last_del_out_bias = del_out_bias
    return last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias, lrnrt_in_to_hid,\
           lrnrt_hid_bias, last_d_in_to_hid, last_d_hid, last_d_hid_to_out, last_d_out, lrnrt_hid_to_out, \
           lrnrt_out_bias, d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias


def bp_full_linear_bias_mmtm_adplrt2(momentum,lrn_max, lrn_min, last_del_in_to_hid, last_d_in_to_hid, lrnrt_in_to_hid, last_del_hid_bias,
                                    last_d_hid, lrnrt_hid_bias, last_del_hid_to_out,  last_d_hid_to_out,
                                    lrnrt_hid_to_out, last_del_out_bias, last_d_out, lrnrt_out_bias,
                                    learning_rate, input_batch, in_to_hid_weights, hid_bias, hid_state,
                                    hid_to_out_weights, out_bias, d_out):
    d_hid_to_out = np.dot(hid_state.transpose(), d_out)
    d_hid = np.dot(d_out, hid_to_out_weights.transpose())
    d_in_to_hid = np.dot(input_batch.transpose(), d_hid)
    d_in = np.dot(d_hid, in_to_hid_weights.transpose())

    multi = (last_d_in_to_hid+np.exp(-30)) * (d_in_to_hid+np.exp(-30))
    flag = multi/np.abs(multi)
    plus = flag * 0.05/2 + 0.05/2
    minus = flag * 0.05/2 + 0.95 + 0.05/2
    lrnrt_in_to_hid += plus
    lrnrt_in_to_hid *= minus

    # maxx = lrnrt_in_to_hid-lrn_max
    # max_one = maxx/np.abs(maxx)
    # max_minus = max_one * 0.05/2 + 0.05/2
    # lrnrt_in_to_hid-=max_minus
    # minn = lrnrt_in_to_hid-lrn_min
    # min_one = minn/np.abs(minn)
    # min_plus = min_one * 0.05/2 + 0.95 + 0.05/2
    # lrnrt_in_to_hid /= min_plus
    #
    multi = (last_d_hid_to_out+np.exp(-30)) * (d_hid_to_out+np.exp(-30))
    flag = multi / np.abs(multi)
    plus = flag * 0.05 / 2 + 0.05 / 2
    minus = flag * 0.05 / 2 + 0.95 + 0.05 / 2
    lrnrt_hid_to_out += plus
    lrnrt_hid_to_out *= minus

    # maxx = lrnrt_hid_to_out - lrn_max
    # max_one = maxx / np.abs(maxx)
    # max_minus = max_one * 0.05 / 2 + 0.05 / 2
    # lrnrt_hid_to_out -= max_minus
    # minn = lrnrt_hid_to_out - lrn_min
    # min_one = minn / np.abs(minn)
    # min_plus = min_one * 0.05 / 2 + 0.95 + 0.05 / 2
    # lrnrt_hid_to_out /= min_plus
    #
    multi = (last_d_hid+np.exp(-30)) * (d_hid+np.exp(-30))
    flag = multi / np.abs(multi)
    plus = flag * 0.05 / 2 + 0.05 / 2
    minus = flag * 0.05 / 2 + 0.95 + 0.05 / 2
    lrnrt_hid_bias += plus
    lrnrt_hid_bias *= minus

    # maxx = lrnrt_hid_bias - lrn_max
    # max_one = maxx / np.abs(maxx)
    # max_minus = max_one * 0.05 / 2 + 0.05 / 2
    # lrnrt_hid_bias -= max_minus
    # minn = lrnrt_hid_bias - lrn_min
    # min_one = minn / np.abs(minn)
    # min_plus = min_one * 0.05 / 2 + 0.95 + 0.05 / 2
    # lrnrt_hid_bias /= min_plus
    #
    multi = (last_d_out+np.exp(-30)) * (d_out+np.exp(-30))
    flag = multi / np.abs(multi)
    plus = flag * 0.05 / 2 + 0.05 / 2
    minus = flag * 0.05 / 2 + 0.95 + 0.05 / 2
    lrnrt_out_bias += plus
    lrnrt_out_bias *= minus

    # maxx = lrnrt_out_bias - lrn_max
    # max_one = maxx / np.abs(maxx)
    # max_minus = max_one * 0.05 / 2 + 0.05 / 2
    # lrnrt_out_bias -= max_minus
    # minn = lrnrt_out_bias - lrn_min
    # min_one = minn / np.abs(minn)
    # min_plus = min_one * 0.05 / 2 + 0.95 + 0.05 / 2
    # lrnrt_out_bias /= min_plus


    # for i in range(0,lrnrt_in_to_hid.shape[0]):
    #     for j in range(0,lrnrt_in_to_hid.shape[1]):
    #         if (last_d_in_to_hid[i][j]*d_in_to_hid[i][j] > 0) & (lrnrt_in_to_hid[i][j] < limit_scale):
    #             lrnrt_in_to_hid += 0.05
    #         if (last_d_in_to_hid[i][j]*d_in_to_hid[i][j] < 0) & (lrnrt_in_to_hid[i][j] > 1/limit_scale):
    #             lrnrt_in_to_hid *= 0.95
    #
    # for i in range(0, lrnrt_hid_to_out.shape[0]):
    #     for j in range(0, lrnrt_hid_to_out.shape[1]):
    #         if (last_d_hid_to_out[i][j] * d_hid_to_out[i][j] > 0) & (lrnrt_hid_to_out[i][j] < limit_scale):
    #             lrnrt_hid_to_out += 0.05
    #         if (last_d_hid_to_out[i][j] * d_hid_to_out[i][j] < 0) & (lrnrt_hid_to_out[i][j] > 1 / limit_scale):
    #             lrnrt_hid_to_out *= 0.95
    #
    # for i in range(0, lrnrt_hid_bias.shape[0]):
    #     for j in range(0, lrnrt_hid_bias.shape[1]):
    #         if (last_d_hid[i][j] * d_hid[i][j] > 0) & (lrnrt_hid_bias[i][j] < limit_scale):
    #             lrnrt_hid_bias += 0.05
    #         if (last_d_hid[i][j] * d_hid[i][j] < 0) & (lrnrt_hid_bias[i][j] > 1 / limit_scale):
    #             lrnrt_hid_bias *= 0.95
    #
    # for i in range(0, lrnrt_out_bias.shape[0]):
    #     for j in range(0, lrnrt_out_bias.shape[1]):
    #         if (last_d_out[i][j] * d_out[i][j] > 0) & (lrnrt_out_bias[i][j] < limit_scale):
    #             lrnrt_out_bias += 0.05
    #         if (last_d_out[i][j] * d_out[i][j] < 0) & (lrnrt_out_bias[i][j] > 1 / limit_scale):
    #             lrnrt_out_bias *= 0.95

    last_d_hid_to_out = d_hid_to_out
    last_d_hid = d_hid
    last_d_in_to_hid = d_in_to_hid
    last_d_out = d_out

    del_hid_to_out = -1 * learning_rate * lrnrt_hid_to_out * d_hid_to_out + momentum * last_del_hid_to_out
    del_in_to_hid = -1 * learning_rate * lrnrt_in_to_hid * d_in_to_hid + momentum * last_del_in_to_hid
    del_hid_bias = -1 * learning_rate * lrnrt_hid_bias * d_hid + momentum * last_del_hid_bias
    del_out_bias = -1 * learning_rate * lrnrt_out_bias * d_out + momentum * last_del_out_bias

    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid
    hid_bias += del_hid_bias
    out_bias += del_out_bias

    last_del_hid_to_out = del_hid_to_out
    last_del_in_to_hid = del_in_to_hid
    last_del_hid_bias = del_hid_bias
    last_del_out_bias = del_out_bias
    return last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias, lrnrt_in_to_hid,\
           lrnrt_hid_bias, last_d_in_to_hid, last_d_hid, last_d_hid_to_out, last_d_out, lrnrt_hid_to_out, \
           lrnrt_out_bias, d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias

def minst_data_processing():
    # data processing
    train_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
                  .iloc[:32000, 1:].values).astype('float32')
    train_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
                   .iloc[:32000, 0].values).astype('int32')
    test_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
                 .iloc[32000:, 1:].values).astype('float32')
    test_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
                  .iloc[32000:, 0].values).astype('int32')
    ori_train_label = train_label
    # one hot encode label
    num_of_labels = train_label.shape[0]
    encoded_labels = np.zeros((num_of_labels, 10),
                              dtype=np.int)
    for i in range(0, num_of_labels):
        encoded_labels[i][train_label[i]] = 1
    train_label = encoded_labels
    print(train_label)
    print(train_label.shape)
    # regularlize data
    train_data /= 255
    test_data /= 255
    return train_data, train_label, test_data, test_label, ori_train_label



