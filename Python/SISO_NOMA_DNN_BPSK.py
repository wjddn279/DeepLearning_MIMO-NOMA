import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

N = 1
M = 1
n_symbol = 100
n_iteration = 1

n_epoch = 1000
modulation_order = 2
SNR_db = np.array([2,4,6,8,10,12,14,16,18,20 ])
batch_size = 100
test_size = 10000

ERROR_user1 = np.zeros([len(SNR_db), n_iteration])

ERROR_user2 = np.zeros([len(SNR_db), n_iteration])

tf.reset_default_graph();
def generate_data(N, M, batch_size):
    a = np.array(range(pow(2, N)))
    K = np.tile(a, batch_size)
    data = K
    
    a = np.eye(modulation_order)
    K = np.tile(a, batch_size)
    label = np.transpose(K)
    return data, label


def generate(M, N, batch_size):
    data,label = generate_data(N, M, batch_size)

    ran1 = np.random.permutation(batch_size * pow(2, N))

    ran2 = np.random.permutation(batch_size * pow(2, N))

    symbol1 = 2 * data[ran1] - 1

    symbol2 = 2 * data[ran2] - 1

    SPC = math.sqrt(0.8) * symbol1 +math.sqrt(0.2) * symbol2

    label1 = label[ran1,:].astype('float32')

    label2 = label[ran2,:].astype('float32')
    
    return SPC, label1, label2


def generate_input(H1_real, H1_image, SPC, N, batch_size, sigma):
    
    N_real, N_image = generate_channel(N, batch_size * pow(2, N), 0)

    input1_real = H1_real * SPC + sigma * N_real

    input1_img = H1_image * SPC + sigma * N_image

    input1 = np.transpose(np.concatenate((input1_real, input1_img), axis=0))

    return input1


def generate_channel(N, M, k):
    h1 = np.random.randn(N, M) / math.sqrt(2)

    h2 = np.random.randn(N, M) / math.sqrt(2)

    if k == 0:
        return h1, h2
    else:
        return 2 * h1, 2 * h2
    
def reciever(input1,num):
    with tf.variable_scope("rx"+str(num)):
        input1 = tf.contrib.layers.batch_norm(input1,center=True, scale=True)
        dense1 = tf.layers.dense(input1, 2*modulation_order, activation=tf.nn.relu)
        dense1 = tf.contrib.layers.batch_norm(dense1,center=True, scale=True)
        dense2 = tf.layers.dense(dense1, 2*modulation_order, activation=tf.nn.relu)
        dense3 = tf.contrib.layers.batch_norm(dense2,center=True, scale=True)
        dense3 = tf.layers.dense(dense3, 2*modulation_order, activation=tf.nn.relu)
        logit1 = tf.contrib.layers.batch_norm(dense3,center=True, scale=True)
        logit1 = tf.layers.dense(dense3, modulation_order)
   
    return logit1


in1 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la1 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

output1 = reciever(in1,1)

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output1, labels=la1))

train_op1 = tf.train.AdamOptimizer(5e-4).minimize(loss1)

acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output1, 1), tf.argmax(la1, 1)),dtype = tf.float32))


in2 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la2 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

output2 = reciever(in2,2)

loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output2, labels=la2))

train_op2 = tf.train.AdamOptimizer(5e-4).minimize(loss2)

acc2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output2, 1), tf.argmax(la2, 1)),dtype = tf.float32))



 
sess = tf.Session()
sess.run(tf.global_variables_initializer())




for k in range(n_iteration):
    H1_real =  0.5286
    H1_image =  -0.1260
    H2_real =  2*H1_real
    H2_image = 2*H1_image
    print ('training iteration %d' %(k))
   
    for i in range(len(SNR_db)):    
                      
        for j in range(n_epoch):
            
            SPC, label1, _ = generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC, 2))
            
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
    
            input1_train = generate_input(H1_real, H1_image, SPC, N, batch_size, sigma)
            
            feed_dict1 = {in1: input1_train, la1: label1}
            accuracy1,los1,out1, _ = sess.run([acc1,loss1,output1,train_op1], feed_dict=feed_dict1)
            
    for i in range(len(SNR_db)):        
             
        for j in range(n_epoch):
        
            SPC, _, label2 = generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC, 2))
            
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
            
            input2_train = generate_input(H2_real, H2_image, SPC, N, batch_size, sigma)
            
            feed_dict2 = {in2: input2_train, la2: label2}
            los2,out2, _ = sess.run([loss2,output2,train_op2], feed_dict=feed_dict2)
            
            
            
for k in range(n_iteration):
    
    print('testing operation %d'%(k))
                    
    for i in range(len(SNR_db)):
            
            
        SPC_test, test_label1, _ = generate(M, N, batch_size * test_size )
        
        signal_power = np.mean(pow(SPC, 2))
            
        sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
        
        input1_test = generate_input(H1_real, H1_image, SPC_test, N, batch_size *test_size, sigma)         
    
        ac1 = sess.run(acc1, feed_dict={in1: input1_test, la1: test_label1})
        
        ERROR_user1[i,k] = 1-ac1
                        
        SPC_test, _, test_label2 = generate(M, N, batch_size * test_size )
        
        signal_power = np.mean(pow(SPC, 2))
            
        sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
        
        input2_test = generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma)
            
        ac2 = sess.run(acc2, feed_dict={in2: input2_test,la2: test_label2})
        
        ERROR_user2[i,k] = 1-ac2   
        

        
error1 = np.mean((ERROR_user1),axis=1)
error2 = np.mean((ERROR_user2),axis=1)

plt.figure()
plt.semilogy(SNR_db,error1, ls = '--', marker = 'o',label='user1')
plt.semilogy(SNR_db,error2, ls = '--', marker = '+',label='user2')
plt.grid()
plt.legend()
plt.ylim(pow(10,-6),pow(10,0) )
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('SER of user2 in SISO_NOMA BPSK_DNN')
plt.savefig('res1')
plt.show()

print(error1)
print(error2)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        