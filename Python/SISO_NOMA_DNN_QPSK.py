import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

N = 1
M = 1
n_symbol = 100
n_iteration = 10

n_epoch = 1000
modulation_order = 4
SNR_db = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])
batch_size = 100
test_size = 10000
ERROR_user1 = np.zeros([len(SNR_db), n_iteration])

ERROR_user2 = np.zeros([len(SNR_db), n_iteration])

tf.reset_default_graph();
def generate_data(N, M, batch_size,modulation_order):
    a = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])/(math.sqrt(2))
    D = np.transpose(np.tile(np.transpose(a),batch_size))
    s1 = D[:,0]
    s2 = D[:,1]
    
    a = np.eye(modulation_order)
    K = np.tile(a, batch_size)
    label = np.transpose(K)
    return s1,s2 ,label


def generate(M, N, batch_size):
    sym_real,sym_imag,label = generate_data(N, M, batch_size,modulation_order)

    ran1 = np.random.permutation(batch_size * pow(2, N))

    ran2 = np.random.permutation(batch_size * pow(2, N))

    sym1_real =  sym_real[ran1] 

    sym1_imag =  sym_imag[ran1] 
    
    sym2_real =  sym_real[ran2] 

    sym2_imag =  sym_imag[ran2]

    SPC_real = math.sqrt(0.8) * sym1_real + math.sqrt(0.2) * sym2_real
    
    SPC_imag = math.sqrt(0.8) * sym1_imag + math.sqrt(0.2) * sym2_imag
    

    label1 = label[ran1].astype('float32')

    label2 = label[ran2].astype('float32')

    return SPC_real,SPC_imag, label1, label2

def generate_input(H1_real, H1_image, SPC_real, SPC_imag, N, batch_size, sigma):
    
    N_real, N_image = generate_channel(N, batch_size * pow(2, N), 0)

    input1_real = H1_real * SPC_real - H1_image * SPC_imag + sigma * N_real

    input1_img = H1_image * SPC_real + H1_real * SPC_imag + sigma * N_image

    input1 = np.transpose(np.concatenate((input1_real, input1_img), axis=0))

    return input1_real,input1_img,input1


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
        dense1 = tf.layers.dense(input1, 3*modulation_order, activation=tf.nn.relu)
        dense1 = tf.contrib.layers.batch_norm(dense1,center=True, scale=True)
        dense2 = tf.layers.dense(dense1, 3*modulation_order, activation=tf.nn.relu)
        dense2 = tf.contrib.layers.batch_norm(dense2,center=True, scale=True)
        dense3 = tf.layers.dense(dense2, 3*modulation_order, activation=tf.nn.relu)
        dense3 = tf.contrib.layers.batch_norm(dense3,center=True, scale=True)
        dense4 = tf.layers.dense(dense3, 3*modulation_order, activation=tf.nn.relu)
        dense4 = tf.contrib.layers.batch_norm(dense4,center=True, scale=True)
        dense5 = tf.layers.dense(dense4, 3*modulation_order, activation=tf.nn.relu)
        dense5 = tf.contrib.layers.batch_norm(dense5,center=True, scale=True)
        dense6 = tf.layers.dense(dense5, 3*modulation_order, activation=tf.nn.relu)
        
        logit1 = tf.layers.dense(dense6, modulation_order,activation=tf.nn.sigmoid)
   
    return logit1


in1 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la1 = tf.placeholder(shape=[None,modulation_order], dtype=tf.float32)

output1 = reciever(in1,1)

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output1, labels=la1))

train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)

acc1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output1, 1), tf.argmax(la1, 1)),dtype = tf.float32))


in2 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la2 = tf.placeholder(shape=[None,modulation_order], dtype=tf.float32)

output2 = reciever(in2,2)

loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output2, labels=la2))

train_op2 = tf.train.AdamOptimizer(1e-4).minimize(loss2)

acc2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output2, 1), tf.argmax(la2, 1)),dtype = tf.float32))



 
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for k in range(n_iteration):
    H1_real, H1_image = -0.4827, - 0.5473
    H2_real, H2_image = -0.7630, - 0.5133
    print ('training iteration %d' %(k))
    
    
    for i in range(len(SNR_db)):    
                      
        for j in range(n_epoch):
            
            SPC_real, SPC_imag, label1, _ = generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC_real, 2)+pow(SPC_imag,2))
            
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
    
            _,_,input1_train = generate_input(H1_real, H1_image, SPC_real,SPC_imag, N, batch_size, sigma)
            
            feed_dict1 = {in1: input1_train, la1: label1}
            accuracy1,los1,out1, _ = sess.run([acc1,loss1,output1,train_op1], feed_dict=feed_dict1)
                    
            "print('n_epoch %d,    loss: %f'%(j + 1, los1))"
             
    
    for i in range(len(SNR_db)):        
             
        for j in range(n_epoch):
        
            SPC_real, SPC_imag, _, label2 = generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC_real, 2)+pow(SPC_imag,2))
            
            sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
    
            _,_,input2_train = generate_input(H2_real, H2_image, SPC_real,SPC_imag, N, batch_size, sigma)
            
            feed_dict2 = {in2: input2_train, la2: label2}
            accuracy2,los2,out2, _ = sess.run([acc2,loss2,output2,train_op2], feed_dict=feed_dict2)
                    
            "print('n_epoch %d,    loss: %f'%(j + 1, los1))"
            
            
for k in range(n_iteration):
    
    print('testing operation %d'%(k))
                    
    for i in range(len(SNR_db)):           
            
                       
        SPC_test_real,SPC_test_imag, test_label1, _ = generate(M, N, batch_size * 1000 )
        
        signal_power = np.mean(pow(SPC_test_real, 2)+pow(SPC_test_imag,2))
            
        sigma = math.sqrt(signal_power / (math.sqrt(N) * pow(10, SNR_db[i] / 10)))
        
        _,_,input1_test = generate_input(H1_real, H1_image, SPC_test_real,SPC_test_imag, N, batch_size * 1000, sigma)         
    
        ac1 = sess.run(acc1, feed_dict={in1: input1_test, la1: test_label1})
        
        ERROR_user1[i,k] = 1-ac1
        
            
        SPC_test_real,SPC_test_imag, _, test_label2 = generate(M, N, batch_size * 1000 )
        
        _,_,input2_test = generate_input(H2_real, H2_image, SPC_test_real,SPC_test_imag, N, batch_size * 1000, sigma)         
    
        ac2 = sess.run(acc2, feed_dict={in2: input2_test, la2: test_label2})
        
        ERROR_user2[i,k] = 1-ac2
        
        
        
error1 = np.mean((ERROR_user1),axis=1)
error2 = np.mean((ERROR_user2),axis=1)

plt.figure()
plt.semilogy(SNR_db,error1, ls = '--', marker = 'o',label='user1')
plt.semilogy(SNR_db,error2, ls = '--', marker = '+',label='user2')
plt.grid()
plt.legend()
plt.ylim(pow(10,-6),0.5)
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('SER of user2 in SISO_NOMA QPSK_DNN')
plt.savefig('res1')
plt.show()


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        