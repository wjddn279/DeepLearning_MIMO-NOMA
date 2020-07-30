import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


class Detector:

    def __init__(self, N, M, SNR_db, n_iteration=1, n_symbol=1000, modulation_order=2, batch_size=5, test_size=1000,
                 learning_rate=1e-4, n_epoch=1000):
        self.N = N
        self.M = M
        self.SNR_db = np.array(SNR_db)
        self.ERROR_user1 = np.zeros([len(SNR_db), n_iteration])
        self.ERROR_user2 = np.zeros([len(SNR_db), n_iteration])
        self.ERROR_user1_ml = np.zeros([len(SNR_db), n_iteration])
        self.ERROR_user2_ml = np.zeros([len(SNR_db), n_iteration])
        self.n_iteration = n_iteration
        self.n_symbol = n_symbol
        self.modulation_order = modulation_order
        self.batch_size = batch_size
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.H1_real, self.H1_image = self.generate_channel(N, M, 0, self.n_iteration)
        self.H2_real, self.H2_image = self.generate_channel(N, M, 1, self.n_iteration)

    def generate_data(self, M, batch_size):
        input_ = [[0 for _ in range(pow(2, M))] for _ in range(M)]
        label = [[0 for _ in range(pow(2, M))] for _ in range(2 * M)]

        for i in range(1 << M):
            for j in range(M):
                if i & (1 << j):
                    input_[M - j - 1][i] = 1

        for i in range(1 << M):
            for j in range(M):
                if input_[j][i] == 1:
                    label[2 * j][i] = 1
                    label[2 * j + 1][i] = 0
                else:
                    label[2 * j][i] = 0
                    label[2 * j + 1][i] = 1

        input_ = np.tile(np.array(input_), batch_size)
        label = np.tile(np.array(label), batch_size)

        return input_, label

    def generate(self, M, N, batch_size):

        data, label = self.generate_data(M, batch_size)

        ran1 = np.random.permutation(batch_size * pow(2, M))  # Suffling Dataset
        ran2 = np.random.permutation(batch_size * pow(2, M))

        symbol1 = 2 * data[:, ran1] - 1
        symbol2 = 2 * data[:, ran2] - 1

        SPC = math.sqrt(0.8) * symbol1 + math.sqrt(0.2) * symbol2  # Superposition Coding

        label1 = np.transpose(label[:, ran1].astype('float32'))
        label2 = np.transpose(label[:, ran2].astype('float32'))

        return SPC, label1, label2, symbol1, symbol2

    def generate_input(self, H1_real, H1_image, SPC, N, batch_size, sigma):

        N_real, N_image = self.generate_channel(N, batch_size * pow(2, N), 0)

        input1_real = np.matmul(H1_real, SPC) + sigma * N_real
        input1_img = np.matmul(H1_image, SPC) + sigma * N_image

        input1 = np.transpose(np.concatenate((input1_real, input1_img), axis=0))

        return input1, input1_real, input1_img

    def generate_channel(self, N, M, k, dimension=0):
        if dimension == 0:
            h1 = np.random.randn(N, M) / math.sqrt(2)
            h2 = np.random.randn(N, M) / math.sqrt(2)
        else:
            h1 = np.random.randn(N, M, dimension) / math.sqrt(2)
            h2 = np.random.randn(N, M, dimension) / math.sqrt(2)
        if k == 0:
            return h1, h2
        else:
            return 2 * h1, 2 * h2

    def reciever(self, input1, num, modulation_order):

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

            logit1 = tf.layers.dense(dense4, modulation_order * self.N)
            logit1 = tf.contrib.layers.batch_norm(logit1, center=True, scale=True)

        return logit1

    def accuracy(self, output, label, M):

        accuracy = 0
        for i in range(M):
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output, [0, 2 * i], [-1, 2]), 1),
                                                  tf.argmax(tf.slice(label, [0, 2 * i], [-1, 2]), 1)),
                                         dtype=tf.float32))

            accuracy += acc

        return accuracy / 8

    def channel_select(self, k)

        return self.H1_real[:, :, k], self.H1_image[:, :, K], self.H2_real[:, :, k], self.H2_image[:, :, k]

    def Maximum_Likelihood(self, a, b, c, d, modulation_order, batch_size, k1, N, test_size):

        a1 = np.repeat(a, pow(modulation_order, 2 * N), axis=1) - np.tile(b,
                                                                          batch_size * test_size * pow(modulation_order,
                                                                                                       N))
        a2 = np.repeat(c, pow(modulation_order, 2 * N), axis=1) - np.tile(d,
                                                                          batch_size * test_size * pow(modulation_order,
                                                                                                       N))
        k3 = np.matmul(np.ones([1, N]), np.sqrt(np.square(a1) + np.square(a2)))
        k4 = np.reshape(k3, [batch_size * test_size * pow(modulation_order, N), pow(modulation_order, 2 * N)])
        k5 = np.argmin(k4, axis=1)
        k6 = k1[:, k5]

        return k6

    def main(self):

        self.Simulation_DNN()
        self.Simulation_ML()

        self.ploting(self.SNR_db)

    def Simulation_DNN(self):

        self.in1 = tf.placeholder(shape=[None, 2 * self.N], dtype=tf.float32)
        self.la1 = tf.placeholder(shape=[None, self.modulation_order * self.N], dtype=tf.float32)
        self.output1 = self.reciever(self.in1, 1, self.modulation_order)
        self.loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output1, labels=self.la1))
        self.train_op1 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss1)
        self.acc1 = self.accuracy(self.output1, self.la1, self.M)

        self.in2 = tf.placeholder(shape=[None, 2 * self.N], dtype=tf.float32)
        self.la2 = tf.placeholder(shape=[None, self.modulation_order * self.N], dtype=tf.float32)
        self.output2 = self.reciever(self.in2, 2, self.modulation_order)
        self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output2, labels=self.la2))
        self.train_op2 = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss2)
        self.acc2 = self.accuracy(self.output2, self.la2, self.M)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for k in range(self.n_iteration):
            tf.reset_default_graph()
            self.training(self.N, self.M, self.SNR_db, self.n_epoch, k, self.batch_size)
            self.testing(self.N, self.M, self.SNR_db, self.test_size, k, self.batch_size)

        self.error1_dnn = np.mean((self.ERROR_user1), axis=1)
        self.error2_dnn = np.mean((self.ERROR_user2), axis=1)

    def training(self, N, M, SNR_db, n_epoch, k, batch_size):

        H1_real, H1_image, H2_real, H2_image = self.channel_select(k)

        print('training iteration %d' % (k))

        for i in range(len(SNR_db)):

            print('User1 , SNR : {}'.format(SNR_db[i]))

            for j in range(n_epoch):
                SPC, label1, _, _, _ = self.generate(M, N, batch_size)
                signal_power = np.mean(pow(SPC, 2))
                sigma_user1 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

                input1_train, _, _ = self.generate_input(H1_real, H1_image, SPC, N, batch_size, sigma_user1)

                feed_dict1 = {self.in1: input1_train, self.la1: label1}
                accuracy1, los1, out1, _ = self.sess.run([self.acc1, self.loss1, self.output1, self.train_op1],
                                                         feed_dict=feed_dict1)

                if j % 100 == 0:
                    print('loss: {}, accuracy: {}'.format(los1, accuracy1))

        for i in range(len(SNR_db)):

            print('User2 , SNR : {}'.format(SNR_db[i]))

            for j in range(n_epoch):
                SPC, _, label2, _, _ = self.generate(M, N, batch_size)
                signal_power = np.mean(pow(SPC, 2))
                sigma_user2 = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

                input2_train, _, _ = self.generate_input(H2_real, H2_image, SPC, N, batch_size, sigma_user2)

                feed_dict2 = {self.in2: input2_train, self.la2: label2}
                accuracy2, los2, out2, _ = self.sess.run([self.acc2, self.loss2, self.output2, self.train_op2],
                                                         feed_dict=feed_dict2)

                if j % 100 == 0:
                    print('loss: {}, accuracy: {}'.format(los2, accuracy2))

    def testing(self, N, M, SNR_db, test_size, k, batch_size):

        H1_real, H1_image, H2_real, H2_image = self.channel_select(k)

        print('testing operation %d' % (k))

        for i in range(len(SNR_db)):
            SPC_test, test_label1, test_label2, symbol1, symbol2 = self.generate(M, N, batch_size * test_size)
            signal_power = np.mean(pow(SPC_test, 2))
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

            input1_test, _, _ = self.generate_input(H1_real, H1_image, SPC_test, N, batch_size * test_size, sigma)

            ac1, out_test1 = self.sess.run([self.acc1, self.output1],
                                           feed_dict={self.in1: input1_test, self.la1: test_label1})
            self.ERROR_user1[i, k] = 1 - ac1

            input2_test, _, _ = self.generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma)

            ac2, out_test2 = self.sess.run([self.acc2, self.output2],
                                           feed_dict={self.in2: input2_test, self.la2: test_label2})
            self.ERROR_user2[i, k] = 1 - ac2

    def simulation_ML(self, N, modulation_order, k, SNR_db, batch_size, test_size):

        H1_real, H1_image, H2_real, H2_image = self.channel_select(k)

        for i in range(len(SNR_db)):
            SPC_test, test_label1, test_label2, symbol1, symbol2 = self.generate(self.M, N, batch_size * test_size)
            signal_power = np.mean(pow(SPC_test, 2))
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, float(SNR_db[i]) / 10.0)))

            _, trans1_real, trans1_img = self.generate_input(H1_real, H1_image, SPC_test, N, pow(modulation_order, N),
                                                        sigma)
            _, trans2_real, trans2_img = self.generate_input(H2_real, H2_image, SPC_test, N, pow(modulation_order, N),
                                                        sigma)

            x_decod_user1 = self.Maximum_Likelihood(trans1_real, np.matmul(H1_real, symbol1), trans1_img, np.matmul(H1_image, symbol1),
                                self.modulation_order, self.batch_size, symbol1, self.N, self.test_size)
            x_decod_user2 = self.Maximum_Likelihood(trans2_real, np.matmul(H2_real, symbol2), trans2_img, np.matmul(H2_image, symbol2),
                               self.modulation_order, self.batch_size, symbol2, self.N, self.test_size)

            self.ERROR_user1_ml[i, k] = np.sum(np.absolute((x_decod_user1 - symbol1) / 2)) / (
                        N * pow(modulation_order, N) * batch_size * test_size)
            self.ERROR_user2_ml[i, k] = np.sum(np.absolute((x_decod_user2 - symbol2) / 2)) / (
                        N * pow(modulation_order, N) * batch_size * test_size)

    def ploting(self, SNR_db):

        self.error1_dnn = np.mean((self.ERROR_user1), axis=1)
        self.error2_dnn = np.mean((self.ERROR_user2), axis=1)

        self.error1_ml = np.mean((self.ERROR_user1_ml), axis=1)
        self.error2_ml = np.mean((self.ERROR_user2_ml), axis=1)

        plt.figure()
        plt.semilogy(SNR_db, self.error1_dnn, ls='--', marker='o', label='user1.dnn')
        plt.semilogy(SNR_db, self.error2_dnn, ls='--', marker='+', label='user2.dnn')
        plt.semilogy(SNR_db, self.error1_ml, marker='o', label='user1.ml')
        plt.semilogy(SNR_db, self.error2_ml, marker='+', label='user2.ml')
        plt.ylim(pow(10, -7), pow(10, 0))
        plt.grid()
        plt.legend()
        plt.xlabel('SNR')
        plt.ylabel('SER')
        plt.title('SER of user2 in 4X4 MIMO_NOMA BPSK')
        plt.savefig('SER_44MIMO_NOMA_DNN_BPSK_dnn_ml')


if __name__ == '__main__':
    SNR_db = range(0, 21, 2)
    Simuation = Detector(8, 8, SNR_db)
    tf.reset_default_graph()
    Simuation.main()