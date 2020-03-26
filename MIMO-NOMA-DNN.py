
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

np.random.seed(5)

N = 8
M = 8
n_symbol = 1000
n_iteration = 1

n_epoch = 1000
modulation_order = 2

SNR_db = np.array(list(range(0 ,11 ,2)))
batch_size = 5
test_size = 1000

ERROR_user1 = np.zeros([len(SNR_db), n_iteration])
ERROR_user2 = np.zeros([len(SNR_db), n_iteration])

tf.reset_default_graph();

def generate_data(M, batch_size):

    input_ = [[0 for _ in range(pow(2, M)) ]for _ in range(M)]
    label = [[0 for _ in range(pow(2, M)) ]for _ in range( 2 *M)]

    for i in range( 1<<M):
        for j in range(M):
            if i & ( 1<<j):
                input_[ M - j -1][i] = 1

    for i in range( 1<<M):
        for j in range(M):
            if input_[j][i] == 1:
                label[ 2 *j][i] = 1
                label[ 2 * j +1][i] = 0
            else:
                label[ 2 *j][i] = 0
                label[ 2 * j +1][i] = 1

    input_ = np.tile(np.array(input_) ,batch_size)
    label =  np.tile(np.array(label) ,batch_size)

    return input_, label

def generate(M, N, batch_size):

    data, label = generate_data(M, batch_size)

    ran1 = np.random.permutation(batch_size * pow(2, M))  # Suffling Dataset
    ran2 = np.random.permutation(batch_size * pow(2, M))

    symbol1 = 2 * data[:, ran1] - 1
    symbol2 = 2 * data[:, ran2] - 1

    SPC = math.sqrt(0.8) * symbol1 + math.sqrt(0.2) * symbol2  # Superposition Coding

    label1 = np.transpose(label[:, ran1].astype('float32'))
    label2 = np.transpose(label[:, ran2].astype('float32'))

    return SPC, label1, label2


def generate_input(H1_real, H1_image, SPC, N, batch_size, sigma):

    N_real, N_image = generate_channel(N, batch_size * pow(2, N), 0)

    input1_real = np.matmul(H1_real, SPC) + sigma * N_real
    input1_img = np.matmul(H1_image, SPC) + sigma * N_image

    input1 = np.transpose(np.concatenate((input1_real, input1_img), axis=0))

    return input1


def generate_channel(N, M, k):

    h1 = np.random.randn(N, M) / math.sqrt(2)
    h2 = np.random.randn(N, M) / math.sqrt(2)

    if k == 0:
        return h1, h2
    else:
        return 2 * h1, 2 * h2


def reciever(input1, num):

    with tf.variable_scope("rx" + str(num)):

        input1 = tf.contrib.layers.batch_norm(input1, center=True, scale=True)
        dense1 = tf.layers.dense(input1, 16 * modulation_order, activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))
        dense1 = tf.contrib.layers.batch_norm(dense1, center=True, scale=True)
        dense2 = tf.layers.dense(dense1, 16 * modulation_order, activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))
        dense2 = tf.contrib.layers.batch_norm(dense2, center=True, scale=True)
        dense3 = tf.layers.dense(dense2, 16 * modulation_order, activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))
        dense3 = tf.contrib.layers.batch_norm(dense3, center=True, scale=True)

        dense4 = tf.layers.dense(dense3, 16 * modulation_order, activation=tf.nn.relu,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.01))
        dense4 = tf.contrib.layers.batch_norm(dense4, center=True, scale=True)

        logit1 = tf.layers.dense(dense4, modulation_order * N)
        logit1 = tf.contrib.layers.batch_norm(logit1, center=True, scale=True)

    return logit1

def accuracy(output ,label ,M):

    accuracy = 0
    for i in range(M):

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output, [0, 2* i], [-1, 2]), 1),
                                              tf.argmax(tf.slice(label, [0, 2 * i], [-1, 2]), 1)),
                                     dtype=tf.float32))

        accuracy += acc

    return accuracy / 8


in1 = tf.placeholder(shape=[None, 2 * N], dtype=tf.float32)
la1 = tf.placeholder(shape=[None, modulation_order * N], dtype=tf.float32)
output1 = reciever(in1, 1)
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output1, labels=la1))
train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)
acc1 = accuracy(output1, la1, M)

in2 = tf.placeholder(shape=[None, 2 * N], dtype=tf.float32)
la2 = tf.placeholder(shape=[None, modulation_order * N], dtype=tf.float32)
output2 = reciever(in2, 2)
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output2, labels=la2))
train_op2 = tf.train.AdamOptimizer(1e-4).minimize(loss2)
acc2 = accuracy(output2, la2, M)

H1_real, H1_image = generate_channel(N, M, 0)
H2_real, H2_image = generate_channel(N, M, 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for k in range(n_iteration):

    print('training iteration %d' % (k))

    H1_real, H1_image = generate_channel(N, M, 0)
    H2_real, H2_image = generate_channel(N, M, 1)

    for i in range(len(SNR_db)):

        print('User1 , SNR : {}'.format(SNR_db[i]))

        for j in range(n_epoch):
            SPC, label1, _ = generate(M, N, batch_size)
            signal_power = np.mean(pow(SPC, 2))
            sigma_user1 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

            input1_train = generate_input(H1_real, H1_image, SPC, N, batch_size, sigma_user1)

            feed_dict1 = {in1: input1_train, la1: label1}
            accuracy1, los1, out1, _ = sess.run([acc1, loss1, output1, train_op1], feed_dict=feed_dict1)

            if j%100 == 0:
                print('loss: {}, accuracy: {}'.format(los1,accuracy1))

    for i in range(len(SNR_db)):

        print('User2 , SNR : {}'.format(SNR_db[i]))

        for j in range(n_epoch):
            SPC, _, label2 = generate(M, N, batch_size)
            signal_power = np.mean(pow(SPC, 2))
            sigma_user2 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

            input2_train = generate_input(H2_real, H2_image, SPC, N, batch_size, sigma_user2)

            feed_dict2 = {in2: input2_train, la2: label2}
            accuracy2,los2, out2, _ = sess.run([acc2,loss2, output2, train_op2], feed_dict=feed_dict2)

            if j%100 == 0:
                print('loss: {}, accuracy: {}'.format(los2,accuracy2))

for k in range(n_iteration):

    print('testing operation %d' % (k))

    for i in range(len(SNR_db)):
        SPC_test, test_label1, test_label2 = generate(M, N, batch_size * test_size)
        signal_power = np.mean(pow(SPC, 2))
        sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

        input1_test = generate_input(H1_real, H1_image, SPC_test, N, batch_size * test_size, sigma)

        ac1, out_test1 = sess.run([acc1, output1], feed_dict={in1: input1_test, la1: test_label1})
        ERROR_user1[i, k] = 1 - ac1

        input2_test = generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma)

        ac2, out_test2 = sess.run([acc2, output2], feed_dict={in2: input2_test, la2: test_label2})
        ERROR_user2[i, k] = 1 - ac2

error1 = np.mean((ERROR_user1), axis=1)
error2 = np.mean((ERROR_user2), axis=1)
print(H1_real)
plt.figure()
plt.semilogy(SNR_db, error1, ls='--', marker='o', label='user1')
plt.semilogy(SNR_db, error2, ls='--', marker='+', label='user2')
plt.grid()
plt.legend()
plt.ylim(pow(10, -6), pow(10, 0))
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('SER of user2 in 4X4 MIMO_NOMA BPSK_DNN')
plt.savefig('SER_44MIMO_NOMA_DNN_BPSK')
plt.show()

print(error1)
print(error2)

