# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time

# data processing
train_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
              .iloc[:32000, 1:].values).astype('float32')
train_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
               .iloc[:32000, 0].values).astype('int32')
test_data = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
             .iloc[32000:, 1:].values).astype('float32')
test_label = (pd.read_csv("~/Developer/kaggle.Digit_Recognizer/datasets/train.csv")
              .iloc[32000:, 0].values).astype('int32')

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


# one layer training


def one_layer_training(epoch, batch_size, alpha, CE_freq, random_seed, num_of_hidden_unit, train_data, train_label,
                       test_data, test_label, weights_init,label):
    # weight init
    rand.seed(a=random_seed)
    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden_unit), dtype=np.float)
    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()
    # print(in_to_hid_weights)
    print(in_to_hid_weights.shape)

    hid_to_out_weights = np.zeros((num_of_hidden_unit,
                                   train_label.shape[1]),
                                  dtype=np.float)
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()



    # train
    start_time = time.time()
    benchmark = np.zeros(1)
    # mini batch fprop
    for i in range(0, epoch):
        train_No = 0;
        batch_index = 0;
        while batch_index + batch_size < train_data.shape[0]:
            train_batch = train_data[batch_index:
            batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:
            batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            # fprop
            hid_state = np.dot(train_batch, in_to_hid_weights)
            out_state = np.dot(hid_state, hid_to_out_weights)
            out_state = 1 / (1 + np.exp(-out_state))

            # bprop:err
            err = out_state - train_label_batch
            CE = (err * err / 2).sum(axis=1)
            d_Out = out_state * (1 - out_state) * err
            d_hid_to_out = np.dot(hid_state.transpose(), d_Out)
            d_hid = np.dot(d_Out, hid_to_out_weights.transpose())
            d_in_to_hid = np.dot(train_batch.transpose(), d_hid)

            # update weights
            del_hid_to_out = -1 * alpha * d_hid_to_out / batch_size
            del_in_to_hid = -1 * alpha * d_in_to_hid / batch_size
            hid_to_out_weights += del_hid_to_out
            in_to_hid_weights += del_in_to_hid

            train_No += 1
            # evaluation

            if train_No % (train_data.shape[0] / batch_size / CE_freq + 1) == 0:
                hid_state = np.dot(test_data, in_to_hid_weights)
                out_state = np.dot(hid_state, hid_to_out_weights)
                out_state = 1 / (1 + np.exp(-out_state))
                prediction = np.argmax(out_state, axis=1)
                evaluation = np.abs(prediction - test_label)
                evaluation = np.bincount(evaluation)[0] / 10000
                benchmark = np.append(benchmark, [evaluation])
                # print(evalu)
                # print(benchmark)
                print(label+' - epoch: ' + str(i) + ' ' + str(train_No * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))

    print('train complete')
    benchmark = benchmark[1:]
    Numbers = range(0, benchmark.shape[0])
    end_time = time.time()

    return Numbers, benchmark, end_time - start_time


def one_layer_training_with_bias(epoch, batch_size, alpha, CE_freq, random_seed, num_of_hidden_unit, train_data,
                                 train_label, test_data, test_label, weights_init,label):
    # weights init with bias
    rand.seed(a=123123)
    # print(rand.random())
    hid_bias = np.ones((batch_size, num_of_hidden_unit), dtype=np.float)
    hid_bias *= weights_init * rand.random()
    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    in_to_hid_weights = np.zeros((train_data.shape[1],
                                  num_of_hidden_unit),
                                 dtype=np.float)
    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()
    # print(in_to_hid_weights)
    print(in_to_hid_weights.shape)

    hid_to_out_weights = np.zeros((num_of_hidden_unit,
                                   train_label.shape[1]),
                                  dtype=np.float)
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()
    # print(hid_to_out_weights)
    # print(hid_to_out_weights.shape)



    ###########train with bias###############
    start_time = time.time()
    benchmark = np.zeros(1)
    # mini batch fprop
    for i in range(0, epoch):
        train_No = 0
        batch_index = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_batch = train_data[batch_index:
            batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:
            batch_index + batch_size, :]
            batch_index = batch_index + batch_size

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

            # update weights
            del_hid_to_out = -1 * alpha * d_hid_to_out / batch_size
            del_in_to_hid = -1 * alpha * d_in_to_hid / batch_size
            hid_to_out_weights += del_hid_to_out
            in_to_hid_weights += del_in_to_hid

            hid_bias += -1 * alpha * d_hid
            out_bias += -1 * alpha * d_Out

            train_No += 1
            # evaluation

            if (train_No * batch_size) % (train_No * batch_size/1000) == 0:
                hid_state = np.dot(test_data, in_to_hid_weights)
                hid_state = hid_state + hid_bias[0]
                out_state = np.dot(hid_state, hid_to_out_weights)
                out_state = out_state + out_bias[0]
                out_state = 1 / (1 + np.exp(-out_state))
                prediction = np.argmax(out_state, axis=1)
                evaluation = np.abs(prediction - test_label)
                evaluation = np.bincount(evaluation)[0] / 10000
                benchmark = np.append(benchmark, [evaluation])
                #print(evaluation)
                # print(benchmark)
                print(label+' - epoch: ' + str(i) + ' ' + str(train_No * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))

    print('train complete')
    benchmark = benchmark[1:]
    Numbers = range(0, benchmark.shape[0])
    # plt.plot(Numbers, benchmark)
    # plt.show()
    end_time = time.time()

    return Numbers, benchmark, end_time - start_time


def one_layer_training_with_bias_api(alpha, train_batch, train_label_batch, in_to_hid_weights, hid_bias,
                                     hid_to_out_weights,out_bias):
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
    del_hid_to_out = -1 * alpha * d_hid_to_out / batch_size
    del_in_to_hid = -1 * alpha * d_in_to_hid / batch_size
    hid_to_out_weights += del_hid_to_out
    in_to_hid_weights += del_in_to_hid

    hid_bias += -1 * alpha * d_hid
    out_bias += -1 * alpha * d_Out

    return in_to_hid_weights,hid_bias,hid_to_out_weights,out_bias,out_state,d_in


def conv_sample_ffbias_training()





#(num1, bench1, eval1, time1) = one_layer_training(5, 3, 0.01, 500, 123123, 200, train_data, train_label, test_data, test_label, 0.01,"no bias")
###hidden neuron
# (num2, bench2, eval2, time2) = one_layer_training_with_bias(5, 3, 0.01, 500, 123123, 1000, train_data, train_label, test_data, test_label, 0.01,"1000 hid")
# (num3, bench3, eval3, time3) = one_layer_training_with_bias(5, 3, 0.01, 500, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"100 hid")
# (num4, bench4, eval4, time4) = one_layer_training_with_bias(5, 3, 0.01, 500, 123123, 10, train_data, train_label, test_data, test_label, 0.01,"10 hid")
# (num5, bench5, eval5, time5) = one_layer_training_with_bias(5, 3, 0.01, 500, 123123, 1, train_data, train_label, test_data, test_label, 0.01,"1 hid")
# print(time2)
# print(time3)
# print(time4)
# print(time5)
# fig, ax = plt.subplots()
# plt.plot(num2, bench2,'b', label="no bias")
# plt.plot(num3, bench3,'g', label="with bias")
# plt.plot(num4, bench4,'y', label="no bias")
# plt.plot(num5, bench5,'r', label="with bias")

(num1, bench1, time1) = one_layer_training_with_bias(5, 1, 0.01, 1000, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"1 batch")
(num2, bench2, time2) = one_layer_training_with_bias(5, 5, 0.01, 1000, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"5 batch")
(num3, bench3, time3) = one_layer_training_with_bias(20, 20, 0.01, 1000, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"20 batch")
(num4, bench4, time4) = one_layer_training_with_bias(100, 100, 0.01, 1000, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"100 batch")
(num5, bench5, time5) = one_layer_training_with_bias(1000, 1000, 0.01, 1000, 123123, 100, train_data, train_label, test_data, test_label, 0.01,"1000 batch")

print(time1)
print(time2)
print(time3)
print(time4)
print(time5)
fig, ax = plt.subplots()
plt.plot(num1, bench1,'k', label="1")
plt.plot(num2, bench2,'b', label="5")
plt.plot(num3, bench3,'g', label="20")
plt.plot(num4, bench4,'y', label="100")
plt.plot(num5, bench5,'r', label="1000")

plt.show()

