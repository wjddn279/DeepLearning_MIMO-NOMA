import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

N = 4
M = 4
n_symbol = 1000
n_iteration = 1

n_epoch = 1000
modulation_order = 2

SNR_db_train = np.array([0])
SNR_db = np.array([0,2,4,6,8,10,12,14,16,18,20])
batch_size = 20
test_size = 5000

ERROR_user1 = np.zeros([len(SNR_db), n_iteration])
ERROR_user2 = np.zeros([len(SNR_db), n_iteration])

ERROR_user1_ml = np.zeros([len(SNR_db), n_iteration])
ERROR_user2_ml = np.zeros([len(SNR_db), n_iteration])


tf.reset_default_graph();

def generate_data(N,M,batch_size):

    M2 = int(M/2)
    e = np.zeros([M,M])
    e[0:M, M2:2*M2] = np.transpose(np.tile(np.eye(M2), M2))
    for i in range(0, M2):
        e[M2 * i:M2 * i + M2, 0:M2] = np.tile(np.eye(M2)[i, 0:M2], (M2, 1))    
    e = e[::-1]
    k = np.zeros([2*M,pow(2,N)])
    for i in range(N):
        b = np.reshape(np.tile((e[i]),M),(M,M))
        c = np.transpose(np.concatenate((b,e),axis=1))
        k[:,4*i:4*i+4] = c
        K = np.tile(k,batch_size)
    label = K
    
    a = np.array(range(pow(2,N)))
    K = np.tile(a,batch_size)
    label_svm = K
    
    
    d = np.zeros([pow(2,M),M])
    for i in range(4):
        a = np.array([[0,0],[0,1],[1,0],[1,1]])
        b = np.reshape(np.tile((a[i]),M),(M,M2))
        c = np.concatenate((b,a),axis=1)
        d[4*i:4*i+4] = c
    D = np.tile(np.transpose(d),batch_size)
    input_ = D
        
    point = 2 *np.transpose(d) - 1
    k2 = np.tile(point,pow(modulation_order,N))
    
    k1 = np.zeros([N,pow(modulation_order,N*2)])
    for i in np.arange(pow(modulation_order,N)):
        p = np.tile(point[:,i],(pow(modulation_order,N)))
        p = np.transpose(np.reshape(p,(pow(modulation_order,N),N)))
        k1[:,16*i:16*(i+1)] = p
    SPC_label = math.sqrt(0.8) * k1 + math.sqrt(0.2) * k2  

    return input_, label, SPC_label,k1,k2,label_svm

def generate(M, N, batch_size):
    
    data,label,SPC_label,k1,k2,label_svm = generate_data(N, M, batch_size)

    ran1 = np.random.permutation(batch_size * pow(2, N))

    ran2 = np.random.permutation(batch_size * pow(2, N))

    symbol1 = 2 * data[:,ran1] - 1

    symbol2 = 2 * data[:,ran2] - 1

    SPC = math.sqrt(0.8) * symbol1 +math.sqrt(0.2) * symbol2

    label1 = np.transpose(label[:,ran1].astype('float32'))

    label2 = np.transpose(label[:,ran2].astype('float32'))
    
    label1_svm = label_svm[ran1].astype('float32')
 
    label2_svm = label_svm[ran2].astype('float32')
    
    return SPC, symbol1,symbol2,label1, label2, SPC_label,k1,k2,label1_svm,label2_svm

def generate_input(H1_real, H1_image, SPC, N, batch_size, sigma):
    
    N_real, N_image = generate_channel(N, batch_size * pow(2, N), 0,0)

    input1_real = np.matmul(H1_real , SPC) + sigma * N_real

    input1_img = np.matmul(H1_image , SPC) + sigma * N_image

    input1 = np.transpose(np.concatenate((input1_real, input1_img), axis=0))

    return input1,input1_real,input1_img

def generate_channel(N, M, k, n_iteration):
    h1 = np.random.randn(N, M, n_iteration) / math.sqrt(2)

    h2 = np.random.randn(N, M, n_iteration) / math.sqrt(2)
    if n_iteration == 0:
        h1 = np.random.randn(N, M) / math.sqrt(2)

        h2 = np.random.randn(N, M) / math.sqrt(2)     

    if k == 0:
        return h1, h2
    else:
        return 2 * h1, 2 * h2
    
def reciever(input1,num):
    with tf.variable_scope("rx"+str(num)):
        input1 = tf.contrib.layers.batch_norm(input1,center=True, scale=True)
        dense1 = tf.layers.dense(input1, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense1 = tf.contrib.layers.batch_norm(dense1,center=True, scale=True)
        dense2 = tf.layers.dense(dense1, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense2 = tf.contrib.layers.batch_norm(dense2,center=True, scale=True)
        dense3 = tf.layers.dense(dense2, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense3 = tf.contrib.layers.batch_norm(dense3,center=True, scale=True)
        dense4 = tf.layers.dense(dense3, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense4 = tf.contrib.layers.batch_norm(dense4,center=True, scale=True)
        dense5 = tf.layers.dense(dense4, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense5 = tf.contrib.layers.batch_norm(dense5,center=True, scale=True)
        dense6 = tf.layers.dense(dense5, 16*modulation_order, activation=tf.nn.relu,kernel_initializer = tf.initializers.truncated_normal(stddev=0.01))
        dense6 = tf.contrib.layers.batch_norm(dense6,center=True, scale=True)
        logit1 = tf.layers.dense(dense6, modulation_order*N)
        logit1 = tf.contrib.layers.batch_norm(logit1,center=True, scale=True)
    return logit1

def ml(a,b,c,d,modulation_order,batch_size,k1):
     
    a1 = np.repeat(a,pow(modulation_order,2*N),axis =1) - np.tile(b,batch_size*test_size*pow(modulation_order,N))
    a2 = np.repeat(c,pow(modulation_order,2*N),axis =1) - np.tile(d,batch_size*test_size*pow(modulation_order,N))
    k3 = np.matmul(np.ones([1,N]),np.sqrt(np.square(a1)+np.square(a2)))
    k4 = np.reshape(k3,[batch_size*test_size*pow(modulation_order,N),pow(modulation_order,2*N)])
    k5 = np.argmin(k4,axis = 1)
    k6 = k1[:,k5]

    return k6


in1 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la1 = tf.placeholder(shape=[None,modulation_order*N], dtype=tf.float32)

output1 = reciever(in1,1)

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output1, labels=la1))

train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)

acc1_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output1,[0,0],[-1,2]), 1),tf.argmax(tf.slice(la1,[0,0],[-1,2]), 1)), dtype = tf.float32))
acc1_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output1,[0,2],[-1,2]), 1),tf.argmax(tf.slice(la1,[0,2],[-1,2]), 1)), dtype = tf.float32))
acc1_3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output1,[0,4],[-1,2]), 1),tf.argmax(tf.slice(la1,[0,4],[-1,2]), 1)), dtype = tf.float32))
acc1_4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output1,[0,6],[-1,2]), 1),tf.argmax(tf.slice(la1,[0,6],[-1,2]), 1)), dtype = tf.float32))
acc1 = (acc1_1+acc1_2+acc1_3+acc1_4)/4

in2 = tf.placeholder(shape=[None,2*N], dtype=tf.float32)

la2 = tf.placeholder(shape=[None,modulation_order*N], dtype=tf.float32)

output2 = reciever(in2,2)

loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output2, labels=la2))

train_op2 = tf.train.AdamOptimizer(1e-4).minimize(loss2)

acc2_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output2,[0,0],[-1,2]), 1),tf.argmax(tf.slice(la2,[0,0],[-1,2]), 1)), dtype = tf.float32))
acc2_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output2,[0,2],[-1,2]), 1),tf.argmax(tf.slice(la2,[0,2],[-1,2]), 1)), dtype = tf.float32))
acc2_3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output2,[0,4],[-1,2]), 1),tf.argmax(tf.slice(la2,[0,4],[-1,2]), 1)), dtype = tf.float32))
acc2_4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.slice(output2,[0,6],[-1,2]), 1),tf.argmax(tf.slice(la2,[0,6],[-1,2]), 1)), dtype = tf.float32))
acc2 = (acc2_1+acc2_2+acc2_3+acc2_4)/4


H1real,H1image = generate_channel(N,M,0,n_iteration)
H2real,H2image = generate_channel(N,M,1,n_iteration)
'''
H1_image = np.array([[-1.544392640192962407e+00	,1.040416436008449796e+00	,-2.553103332841635820e-01	,5.462819251163659739e-01],[-5.245457267467228402e-01,	5.784440450256107535e-01,	-3.356213119571275771e-01,	-1.424159231325177632e+00],[-8.125053763411869134e-01,	-5.852300542731407873e-01,	-3.601322440681380965e-01,	-1.744946382711259902e-01],[7.881593467856291557e-01,	6.081413620848292734e-01,	7.701683825511492776e-01,	-1.212763846772349607e+00]])

H1_real = np.array([[1.414423548142727416e+00,	4.273564551036619363e-01,	-6.146789170238127209e-01	,5.454669563977698754e-01],
[-7.962996677894614017e-02	,8.181490209622260812e-02,	7.959509966772851941e-01,	2.298539366398068828e-01],
[-1.712577253895157803e+00	,-1.096081382309416208e-01	,3.565677907317481576e-01,	-6.247398122905933882e-01],
[-1.737353773876517404e-01	,-1.199654059015842905e-01,	-6.251809249583253347e-01,	-5.857909959840443825e-01]])
H2_real =np.array([[8.357156417270315829e-02,	1.357685907252838664e+00,	-6.767576470919008935e-01	,-5.662564137116241070e-02],
[1.781247977199322541e-01,	8.462328558123506372e-01,	-2.101680109000647612e+00	,5.046216545767419175e-03],
[9.674915696750956418e-01	,-1.308457047131135198e+00	,-1.431659875679342386e+00	,-4.087945593252304000e-01],
[2.996723246207782410e+00	,-1.206807601551023623e+00	,-6.267314197164964851e-01	,1.512265748505495222e-01]])

H2_image =np.array([[-2.220988895338463998e+00	,-1.319153835851952117e+00,	1.176845844863776192e+00,	-2.813816225614304556e-02],
[-5.789725239721320582e-01	,5.699508374843409442e-01	,-1.565048691368789546e+00	,-1.643232410716657932e-01],
[-9.784310516014633752e-01,	4.280551250734762614e-01,	1.006788556710295657e+00	,2.845221723119386592e+00],
[8.558006284710916178e-01,	-1.500063807363221313e+00	,-1.799920597873126527e+00	,-2.567973128197103438e+00]])
'''

H1_real = np.array([[0.7565  ,  0.0246 ,   0.4708 ,  -0.6736],
   [-1.3732 ,   0.7148   , 0.5825 ,   0.7321],
  [ -0.8394  , -1.3591  , -0.0328  ,  0.4201],
   [ 0.4125   , 0.3928 ,   0.3829   ,-0.9459]])
H1_image = np.array([[   -0.9027  ,  0.3607 ,   1.7073 ,   1.4104],
    [0.2397  ,  1.2623  , -0.0489 ,   0.7416],
    [0.6492   ,-1.5532  , -0.0702  , -2.4573],
    [1.3094   , 0.5485 ,   1.0087  , -0.2718]])
H2_real = np.array([[    2.0915 ,  -0.0277 ,  -2.8472 ,   0.3049],
   [-1.2719  , -0.9617  , -0.8682  ,  1.3288],
    [1.1487  ,  0.9615  , -0.5566  , -2.5416],
    [1.5289  , -3.2012 ,  -0.3904  ,  1.0534]])
H2_image = np.array([[    2.9684  , -0.1825 ,  -0.3127 ,   1.0636],
   [-0.2159 ,   0.7222  ,  1.0957 ,  -0.9675],
   [-1.6677  , -1.0497  ,  0.1018  , -0.0674],
    [0.7354  ,  0.0341 ,   1.8606  , -0.1552]])

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

for k in range(n_iteration):

    print ('training iteration %d' %(k))
    
    H1_real = H1real[:,:,k]
    H1_image = H1image[:,:,k]
    H2_real = H2real[:,:,k]
    H2_image = H2image[:,:,k]
    

    for i in range(len(SNR_db_train)):    
        
        print('training SNR_db %d user1' %(SNR_db_train[i]))
        
                      
        for j in range(n_epoch):
            
            SPC,_,_,label1, _,_ ,_,_,label1_svm,_= generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC, 2))
            
            sigma = math.sqrt(signal_power* math.sqrt(N)  /  pow(10, float(SNR_db_train[i]) / 10.0))
    
            input1_train,_,_ = generate_input(H1_real, H1_image, SPC, N, batch_size, sigma)
            
            feed_dict1 = {in1: input1_train, la1: label1}
            accuracy1,los1,out1, _ = sess.run([acc1,loss1,output1,train_op1], feed_dict=feed_dict1)
                       
            
    for i in range(len(SNR_db_train)):       
        
        print('training SNR_db %d user2' %(SNR_db_train[i]))
             
        for j in range(n_epoch):
        
            SPC,_,_,_, label2,_ ,_,_,_,label2_svm= generate(M, N, batch_size)
    
            signal_power = np.mean(pow(SPC, 2))
            
            sigma = math.sqrt(signal_power* math.sqrt(N)  /  pow(10, float(SNR_db_train[i]) / 10.0))
            
            input2_train,_,_ = generate_input(H2_real, H2_image, SPC, N, batch_size, sigma)
            
            feed_dict2 = {in2: input2_train, la2: label2}
            los2,out2, _ = sess.run([loss2,output2,train_op2], feed_dict=feed_dict2)
            
            
             
    for i in range(len(SNR_db)):
        
        print('testing SNR_db %d ' %(SNR_db[i]))
                        
        SPC_test, symbol1,symbol2,test_label1, test_label2, SPC_label,sym1,sym2,svm_test1,svm_test2 = generate(M, N, batch_size * test_size )
        
        signal_power = np.mean(pow(SPC_test, 2))
            
        sigma = math.sqrt(signal_power* math.sqrt(N) /   pow(10, float(SNR_db[i]) / 10.0))
        
        input1_test,trans1_real,trans1_img = generate_input(H1_real, H1_image, SPC_test, N, batch_size *test_size, sigma)         
        input2_test,trans2_real,trans2_img = generate_input(H2_real, H2_image, SPC_test, N, batch_size * test_size, sigma)
            
        "Deep learning network"
        ac1,out_test1 = sess.run([acc1,output1], feed_dict={in1: input1_test, la1: test_label1})
        
        ERROR_user1[i,k] = 1-ac1                                       
    
        ac2,out_test2 = sess.run([acc2,output2], feed_dict={in2: input2_test,la2: test_label2})
        
        ERROR_user2[i,k] = 1-ac2   
        
        "Support Vector Machine"
        
        
        "MAXIMUM LIKELIHOOD DETECTION"
        _,label1_real,label1_img = generate_input(H1_real, H1_image, SPC_label, N, pow(modulation_order,N), 0)
        _,label2_real,label2_img = generate_input(H2_real, H2_image, SPC_label, N, pow(modulation_order,N), 0)
        
        "Detection for user1"
        
        x_decod_user1 = ml(trans1_real,label1_real,trans1_img,label1_img,modulation_order,batch_size,sym1)
        x_decod_user2 = ml(trans2_real,label2_real,trans2_img,label2_img,modulation_order,batch_size,sym2)
         
        ERROR_user1_ml[i,k] = np.sum(np.absolute((x_decod_user1-symbol1)/2))/(N*pow(modulation_order,N)*batch_size*test_size)
        ERROR_user2_ml[i,k] = np.sum(np.absolute((x_decod_user2-symbol2)/2))/(N*pow(modulation_order,N)*batch_size*test_size)
        
    
error1_dnn = np.mean((ERROR_user1),axis=1)
error2_dnn = np.mean((ERROR_user2),axis=1)

error1_ml = np.mean((ERROR_user1_ml),axis=1)
error2_ml = np.mean((ERROR_user2_ml),axis=1)


plt.figure()
plt.semilogy(SNR_db,error1_dnn, ls = '--', marker = 'o',label='user1.dnn')
plt.semilogy(SNR_db,error2_dnn, ls = '--', marker = '+',label='user2.dnn')
plt.semilogy(SNR_db,error1_ml, marker = 'o',label='user1.ml')
plt.semilogy(SNR_db,error2_ml, marker = '+',label='user2.ml')
plt.ylim(pow(10,-7),pow(10,0) )
plt.grid()
plt.legend()
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('SER of user2 in 4X4 MIMO_NOMA BPSK')
plt.savefig('SER_44MIMO_NOMA_DNN_BPSK_dnn_ml')


print(error1_dnn)
print(error2_dnn)
print(error1_ml)
print(error2_ml)



