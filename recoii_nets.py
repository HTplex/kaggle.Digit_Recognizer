import recoii_props as rp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import time


def in_hid_sigmoid_out_bias(num_of_epoch, batch_size, learning_rate, test_freq, weights_init,
                            random_seed, num_of_hidden, label):

    (train_data, train_label, test_data, test_label,ori_test_label) = rp.minst_data_processing()
    print("data imported")
    # weight init
    rand.seed(a=random_seed)
    hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    hid_bias *= weights_init * rand.random()
    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    hid_to_out_weights = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)

    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()
    print("w init complete")
    #train:
    start_time = time.time()
    benchmark = [0]
    bench_out_state = [0]
    for epoch_no in range(0,num_of_epoch):
        batch_index = 0
        train_no = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_no += 1
            train_batch = train_data[batch_index:batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            (hid_state, out_state) = rp.fp_full_linear_bias(train_batch, in_to_hid_weights, hid_bias, hid_to_out_weights
                                                            , out_bias)
            out_state = 1 / (1 + np.exp(-out_state))
            error = out_state - train_label_batch
            d_out = out_state * (1 - out_state) * error
            (d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias) = \
                rp.bp_full_linear_bias(learning_rate,train_batch, in_to_hid_weights, hid_bias, hid_state,
                                       hid_to_out_weights,out_bias, d_out)

            #benchmark
            if (train_no * batch_size) % (train_no * batch_size / test_freq) == 0:
                (bench_hid_state, bench_out_state) = rp.fp_full_linear_bias(test_data, in_to_hid_weights, hid_bias[0],
                                                                            hid_to_out_weights, out_bias[0])
                bench_out_state = 1 / (1 + np.exp(-bench_out_state))
                prediction = np.argmax(bench_out_state, axis=1)
                evaluation = np.abs(prediction - test_label)
                evaluation = np.bincount(evaluation)[0] / test_label.shape[0]
                benchmark = np.append(benchmark, [evaluation])
                print(label + ' - epoch: ' + str(epoch_no) + ' ' + str(
                    train_no * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))
    print("done")
    benchmark = benchmark[1:]
    numbers = range(0, benchmark.shape[0])
    plt.plot(numbers, benchmark)
    plt.show()
    end_time = time.time()

    return benchmark, end_time - start_time


def in_hid_sigmoid_out_bias_mmtm(momentum, num_of_epoch, batch_size, learning_rate, test_freq, weights_init, random_seed,
                                 num_of_hidden, label):

    (train_data, train_label, test_data, test_label,ori_test_label) = rp.minst_data_processing()
    print("data imported")
    # weight init
    rand.seed(a=random_seed)
    hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    hid_bias *= weights_init * rand.random()
    last_del_hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    last_del_out_bias = np.ones((batch_size, 10), dtype=np.float)
    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_del_in_to_hid = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    hid_to_out_weights = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_del_hid_to_out = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)

    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()
    print("w init complete")
    #train:
    start_time = time.time()
    benchmark = [0]
    bench_out_state = [0]
    for epoch_no in range(0,num_of_epoch):
        batch_index = 0
        train_no = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_no += 1
            train_batch = train_data[batch_index:batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            (hid_state, out_state) = rp.fp_full_linear_bias(train_batch, in_to_hid_weights, hid_bias, hid_to_out_weights
                                                            , out_bias)
            out_state = 1 / (1 + np.exp(-out_state))
            error = out_state - train_label_batch
            d_out = out_state * (1 - out_state) * error
            (last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias,d_in, in_to_hid_weights,
             hid_bias, hid_to_out_weights, out_bias) = \
                rp.bp_full_linear_bias_mmtm(momentum, last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out,  last_del_out_bias,
                             learning_rate, train_batch, in_to_hid_weights, hid_bias, hid_state, hid_to_out_weights,
                             out_bias, d_out)

            #benchmark
            if (train_no * batch_size) % (train_data.shape[0]/ test_freq) == 0:
                (bench_hid_state, bench_out_state) = rp.fp_full_linear_bias(test_data, in_to_hid_weights, hid_bias[0],
                                                                            hid_to_out_weights, out_bias[0])
                bench_out_state = 1 / (1 + np.exp(-bench_out_state))
                prediction = np.argmax(bench_out_state, axis=1)
                rightmap = np.abs(prediction - test_label)
                evaluation = np.bincount(rightmap)[0] / test_label.shape[0]
                benchmark = np.append(benchmark, [evaluation])
                print(label + ' - epoch: ' + str(epoch_no) + ' ' + str(
                    train_no * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))
    print("done")
    benchmark = benchmark[1:]
    numbers = range(0, benchmark.shape[0])
    plt.plot(numbers, benchmark)
    plt.show()
    end_time = time.time()

    return benchmark, end_time - start_time, rightmap,prediction


def in_hid_softmax_out_bias_mmtm(momentum, num_of_epoch, batch_size, learning_rate, test_freq, weights_init, random_seed,
                                 num_of_hidden, label):

    (train_data, train_label, test_data, test_label,ori_test_label) = rp.minst_data_processing()
    print("data imported")
    # weight init
    rand.seed(a=random_seed)
    hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    hid_bias *= weights_init * rand.random()
    last_del_hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    last_del_out_bias = np.ones((batch_size, 10), dtype=np.float)
    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_del_in_to_hid = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    hid_to_out_weights = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_del_hid_to_out = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)

    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()
    print("w init complete")
    #train:
    start_time = time.time()
    benchmark = [0]
    bench_out_state = [0]
    for epoch_no in range(0,num_of_epoch):
        batch_index = 0
        train_no = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_no += 1
            train_batch = train_data[batch_index:batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            (hid_state, out_state) = rp.fp_full_linear_bias(train_batch, in_to_hid_weights, hid_bias, hid_to_out_weights
                                                            , out_bias)
            out_state_exp = np.exp(out_state)
            #print(out_state_exp.shape)
            out_state_sum = np.sum(out_state_exp,axis=1)
            out_state_sum = out_state_sum.reshape(out_state_sum.shape[0],1)
            #print(out_state_sum.shape)
            ones = np.zeros((1, out_state.shape[1]), dtype=np.float)
            ones.fill(1)
            out_state_sum_expand = np.dot(out_state_sum,ones)
            out_state_softmax = out_state_exp/out_state_sum_expand
            #error = out_state - train_label_batch
            d_out = out_state_softmax-train_label_batch
            (last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias,d_in, in_to_hid_weights,
             hid_bias, hid_to_out_weights, out_bias) = \
                rp.bp_full_linear_bias_mmtm(momentum, last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out,  last_del_out_bias,
                             learning_rate, train_batch, in_to_hid_weights, hid_bias, hid_state, hid_to_out_weights,
                             out_bias, d_out)

            #benchmark
            if (train_no * batch_size) % (train_data.shape[0]/ test_freq) == 0:
                (bench_hid_state, bench_out_state) = rp.fp_full_linear_bias(test_data, in_to_hid_weights, hid_bias[0],
                                                                            hid_to_out_weights, out_bias[0])
                bench_out_state = 1 / (1 + np.exp(-bench_out_state))
                prediction = np.argmax(bench_out_state, axis=1)
                rightmap = np.abs(prediction - test_label)
                evaluation = np.bincount(rightmap)[0] / test_label.shape[0]
                benchmark = np.append(benchmark, [evaluation])
                print(label + ' - epoch: ' + str(epoch_no) + ' ' + str(
                    train_no * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))
    print("done")
    benchmark = benchmark[1:]
    numbers = range(0, benchmark.shape[0])
    plt.plot(numbers, benchmark)
    plt.show()
    end_time = time.time()
    np.savetxt("wrong.csv", prediction, delimiter=',')

    return benchmark, end_time - start_time


def in_hid_softmax_out_bias_mmtm_adplrt(momentum,limit_max,limit_min, num_of_epoch, batch_size, learning_rate, test_freq, weights_init, random_seed,
                                 num_of_hidden, label):

    (train_data, train_label, test_data, test_label,ori_test_label) = rp.minst_data_processing()
    print("data imported")
    # weight init
    rand.seed(a=random_seed)
    hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    hid_bias *= weights_init * rand.random()
    last_del_hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)
    last_d_hid = np.ones((batch_size, num_of_hidden), dtype=np.float)
    lrnrt_hid_bias = np.ones((batch_size, num_of_hidden), dtype=np.float)

    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    last_del_out_bias = np.ones((batch_size, 10), dtype=np.float)
    last_d_out = np.ones((batch_size, 10), dtype=np.float)
    lrnrt_out_bias = np.ones((batch_size, 10), dtype=np.float)

    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_del_in_to_hid = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_d_in_to_hid = np.ones((train_data.shape[1], num_of_hidden), dtype=np.float)
    lrnrt_in_to_hid = np.ones((train_data.shape[1], num_of_hidden), dtype=np.float)

    hid_to_out_weights = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_del_hid_to_out = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_d_hid_to_out = np.ones((num_of_hidden, train_label.shape[1]), dtype=np.float)
    lrnrt_hid_to_out = np.ones((num_of_hidden, train_label.shape[1]), dtype=np.float)

    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()+weights_init
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()+weights_init
    print("w init complete")
    #train:
    start_time = time.time()
    benchmark = [0]
    bench_out_state = [0]
    for epoch_no in range(0,num_of_epoch):
        batch_index = 0
        train_no = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_no += 1
            train_batch = train_data[batch_index:batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            (hid_state, out_state) = rp.fp_full_linear_bias(train_batch, in_to_hid_weights, hid_bias, hid_to_out_weights
                                                            , out_bias)
            out_state_exp = np.exp(out_state)
            #print(out_state_exp.shape)
            out_state_sum = np.sum(out_state_exp,axis=1)
            out_state_sum = out_state_sum.reshape(out_state_sum.shape[0],1)
            #print(out_state_sum.shape)
            ones = np.zeros((1, out_state.shape[1]), dtype=np.float)
            ones.fill(1)
            out_state_sum_expand = np.dot(out_state_sum,ones)
            out_state_softmax = out_state_exp/out_state_sum_expand
            #error = out_state - train_label_batch
            d_out = out_state_softmax-train_label_batch
            (last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias, lrnrt_in_to_hid,
           lrnrt_hid_bias, last_d_in_to_hid, last_d_hid, last_d_hid_to_out, last_d_out, lrnrt_hid_to_out,
           lrnrt_out_bias, d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias) = \
                rp.bp_full_linear_bias_mmtm_adplrt2(momentum,limit_max,limit_min, last_del_in_to_hid, last_d_in_to_hid, lrnrt_in_to_hid, last_del_hid_bias,
                                    last_d_hid, lrnrt_hid_bias, last_del_hid_to_out,  last_d_hid_to_out,
                                    lrnrt_hid_to_out, last_del_out_bias, last_d_out, lrnrt_out_bias,
                                    learning_rate, train_batch, in_to_hid_weights, hid_bias, hid_state,
                                    hid_to_out_weights, out_bias, d_out)

            #benchmark
            if (train_no * batch_size) % (train_data.shape[0]/ test_freq) == 0:
                (bench_hid_state, bench_out_state) = rp.fp_full_linear_bias(test_data, in_to_hid_weights, hid_bias[0],
                                                                            hid_to_out_weights, out_bias[0])
                bench_out_state = 1 / (1 + np.exp(-bench_out_state))
                prediction = np.argmax(bench_out_state, axis=1)
                rightmap = np.abs(prediction - test_label)
                evaluation = np.bincount(rightmap)[0] / test_label.shape[0]
                benchmark = np.append(benchmark, [evaluation])
                print(label + ' - epoch: ' + str(epoch_no) + ' ' + str(
                    train_no * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))
    print("done")
    benchmark = benchmark[1:]
    numbers = range(0, benchmark.shape[0])
    plt.plot(numbers, benchmark)
    plt.show()
    end_time = time.time()
    np.savetxt("wrong.csv", prediction, delimiter=',')

    return benchmark, end_time - start_time
#def in_hid_sigmoid_out_bias_mmtm_
#(ben1,t1) = in_hid_sigmoid_out_bias(1, 1, 0.01, 1000, 0.01, 123123, 200,"test")


def in_hid_hid_softmax_out_bias_mmtm_adplrt(momentum,limit_max,limit_min, num_of_epoch, batch_size, learning_rate, test_freq, weights_init, random_seed,
                                 num_of_hidden1, num_of_hidden2, label):

    (train_data, train_label, test_data, test_label,ori_test_label) = rp.minst_data_processing()
    print("data imported")
    # weight init
    rand.seed(a=random_seed)
    #biases
    hid1_bias = np.ones((batch_size, num_of_hidden1), dtype=np.float)
    hid1_bias *= weights_init * rand.random()
    last_del_hid1_bias = np.ones((batch_size, num_of_hidden1), dtype=np.float)
    last_d_hid1 = np.ones((batch_size, num_of_hidden1), dtype=np.float)
    lrnrt_hid1_bias = np.ones((batch_size, num_of_hidden1), dtype=np.float)

    hid2_bias = np.ones((batch_size, num_of_hidden2), dtype=np.float)
    hid2_bias *= weights_init * rand.random()
    last_del_hid2_bias = np.ones((batch_size, num_of_hidden2), dtype=np.float)
    last_d_hid2 = np.ones((batch_size, num_of_hidden2), dtype=np.float)
    lrnrt_hid2_bias = np.ones((batch_size, num_of_hidden2), dtype=np.float)

    out_bias = np.ones((batch_size, 10), dtype=np.float)
    out_bias *= weights_init * rand.random()
    last_del_out_bias = np.ones((batch_size, 10), dtype=np.float)
    last_d_out = np.ones((batch_size, 10), dtype=np.float)
    lrnrt_out_bias = np.ones((batch_size, 10), dtype=np.float)

    #weights
    in_to_hid_weights = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_del_in_to_hid = np.zeros((train_data.shape[1], num_of_hidden), dtype=np.float)
    last_d_in_to_hid = np.ones((train_data.shape[1], num_of_hidden), dtype=np.float)
    lrnrt_in_to_hid = np.ones((train_data.shape[1], num_of_hidden), dtype=np.float)

    hid_to_out_weights = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_del_hid_to_out = np.zeros((num_of_hidden, train_label.shape[1]), dtype=np.float)
    last_d_hid_to_out = np.ones((num_of_hidden, train_label.shape[1]), dtype=np.float)
    lrnrt_hid_to_out = np.ones((num_of_hidden, train_label.shape[1]), dtype=np.float)

    for i in range(0, in_to_hid_weights.shape[0]):
        for j in range(0, in_to_hid_weights.shape[1]):
            in_to_hid_weights[i][j] = weights_init * rand.random()+weights_init
    for i in range(0, hid_to_out_weights.shape[0]):
        for j in range(0, hid_to_out_weights.shape[1]):
            hid_to_out_weights[i][j] = weights_init * rand.random()+weights_init
    print("w init complete")
    #train:
    start_time = time.time()
    benchmark = [0]
    bench_out_state = [0]
    for epoch_no in range(0,num_of_epoch):
        batch_index = 0
        train_no = 0
        while batch_index + batch_size < train_data.shape[0]:
            train_no += 1
            train_batch = train_data[batch_index:batch_index + batch_size, :]
            train_label_batch = train_label[batch_index:batch_index + batch_size, :]
            batch_index = batch_index + batch_size

            (hid_state, out_state) = rp.fp_full_linear_bias(train_batch, in_to_hid_weights, hid_bias, hid_to_out_weights
                                                            , out_bias)
            out_state_exp = np.exp(out_state)
            #print(out_state_exp.shape)
            out_state_sum = np.sum(out_state_exp,axis=1)
            out_state_sum = out_state_sum.reshape(out_state_sum.shape[0],1)
            #print(out_state_sum.shape)
            ones = np.zeros((1, out_state.shape[1]), dtype=np.float)
            ones.fill(1)
            out_state_sum_expand = np.dot(out_state_sum,ones)
            out_state_softmax = out_state_exp/out_state_sum_expand
            #error = out_state - train_label_batch
            d_out = out_state_softmax-train_label_batch
            (last_del_in_to_hid, last_del_hid_bias, last_del_hid_to_out, last_del_out_bias, lrnrt_in_to_hid,
           lrnrt_hid_bias, last_d_in_to_hid, last_d_hid, last_d_hid_to_out, last_d_out, lrnrt_hid_to_out,
           lrnrt_out_bias, d_in, in_to_hid_weights, hid_bias, hid_to_out_weights, out_bias) = \
                rp.bp_full_linear_bias_mmtm_adplrt2(momentum,limit_max,limit_min, last_del_in_to_hid, last_d_in_to_hid, lrnrt_in_to_hid, last_del_hid_bias,
                                    last_d_hid, lrnrt_hid_bias, last_del_hid_to_out,  last_d_hid_to_out,
                                    lrnrt_hid_to_out, last_del_out_bias, last_d_out, lrnrt_out_bias,
                                    learning_rate, train_batch, in_to_hid_weights, hid_bias, hid_state,
                                    hid_to_out_weights, out_bias, d_out)

            #benchmark
            if (train_no * batch_size) % (train_data.shape[0]/ test_freq) == 0:
                (bench_hid_state, bench_out_state) = rp.fp_full_linear_bias(test_data, in_to_hid_weights, hid_bias[0],
                                                                            hid_to_out_weights, out_bias[0])
                bench_out_state = 1 / (1 + np.exp(-bench_out_state))
                prediction = np.argmax(bench_out_state, axis=1)
                rightmap = np.abs(prediction - test_label)
                evaluation = np.bincount(rightmap)[0] / test_label.shape[0]
                benchmark = np.append(benchmark, [evaluation])
                print(label + ' - epoch: ' + str(epoch_no) + ' ' + str(
                    train_no * batch_size / (train_data.shape[0] / 100)) + '% :: '
                      + str(evaluation))
    print("done")
    benchmark = benchmark[1:]
    numbers = range(0, benchmark.shape[0])
    plt.plot(numbers, benchmark)
    plt.show()
    end_time = time.time()
    np.savetxt("wrong.csv", prediction, delimiter=',')

    return benchmark, end_time - start_time


#(ben2,t2, rmap,pdt) = in_hid_sigmoid_out_bias_mmtm(0.1, 3, 1, 0.05, 20, 0.01, 123123, 1000,"testm")
#(ben3,t3) = in_hid_softmax_out_bias_mmtm(0.3, 100, 5, 0.01, 10, 0.01, 123123, 1000, "softmax")
#(ben4,t4) = in_hid_softmax_out_bias_mmtm_adplrt(0,100, 100, 5, 0.01, 1000, 0.01, 123123, 100, "adaptive lrt")
(ben5, t5) = in_hid_softmax_out_bias_mmtm_adplrt(0,100,0.000001, 100, 200, 0.0005, 100, 0.01, 123123, 200, "adaptive lrt2")
# print(rmap.shape)
# for i in range(0,rmap.shape[0]):
#     if(rmap[i]!=0):
#         print(str(i))
# print(pdt)
print(rand.random())

