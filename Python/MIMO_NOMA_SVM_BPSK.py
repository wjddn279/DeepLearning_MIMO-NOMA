import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
N = 4
M = 4
n_symbol = 100
n_iteration = 1

modulation_order = 2
SNR_db = np.array([2,4,6,8,10,12,14,16,18,20])
batch_size = 1000
test = 100

ERROR_user1 = np.zeros([len(SNR_db),n_iteration])

ERROR_user2 = np.zeros([len(SNR_db),n_iteration])

def generate_data(N,M,batch_size):
    M2 = int(M/2)
    a = np.array(range(pow(2,N)))
    K = np.tile(a,batch_size)
    label = K


    d = np.zeros([pow(2,M),M])
    for i in range(4):
        a = np.array([[0,0],[0,1],[1,0],[1,1]])
        b = np.reshape(np.tile((a[i]),M),(M,M2))
        c = np.concatenate((b,a),axis=1)
        d[4*i:4*i+4] = c
    D = np.tile(np.transpose(d),batch_size)
    input_ = D
    
    return input_, label

def generate(M,N,batch_size):
    
    data,label = generate_data(N,M,batch_size)
    
    ran1 = np.random.permutation(batch_size*pow(2,N))

    ran2 = np.random.permutation(batch_size*pow(2,N))

    symbol1 = 2 * data[:,ran1] - 1

    symbol2 = 2 * data[:,ran2] - 1

    SPC = math.sqrt(0.8) * symbol1 + math.sqrt(0.2) * symbol2
    
    label1 = label[ran1].astype('float32')

    label2 = label[ran2].astype('float32')
    
    return SPC,label1,label2

    
def generate_input(H1_real,H1_image,SPC,N,batch_size,sigma):
    
    N_real, N_image = generate_channel(N,batch_size*pow(2,N),0)
 
    input1_real = np.matmul(H1_real,SPC) + sigma * N_real

    input1_img = np.matmul(H1_image,SPC) + sigma * N_image
        
    input1 = np.transpose(np.concatenate((input1_real, input1_img), axis = 0))
    
    return input1


 
def generate_channel(N,M,k):

    h1 = np.random.randn(N, M) / math.sqrt(2)

    h2 = np.random.randn(N, M) / math.sqrt(2)
    
    if k==0:
        return h1,h2
    else:
        return 2*h1,2*h2



H1_real ,H1_image = generate_channel(N,M,0)
H2_real, H2_image = generate_channel(N, M,1)


svm_user2 = SVC(gamma='auto')
svm_user1 = SVC(gamma='auto')




for j in range(n_iteration) :
    
    SPC,label1,label2 = generate(M,N,batch_size)
    
    
    signal_power = np.mean(pow(SPC,2))

    print (j)

    for i in range(len(SNR_db)) :        
        
        sigma = math.sqrt(signal_power/(math.sqrt(N)*pow(10,SNR_db[i]/10)))
        
        input1_train = generate_input(H1_real,H1_image,SPC,N,batch_size,sigma)
        
        input2_train = generate_input(H2_real,H2_image,SPC,N,batch_size,sigma)

        svm_user2.fit(input2_train, label2)
        svm_user1.fit(input1_train, label1)

        

        

        
        SPC_test,test_label1,test_label2 = generate(M,N,batch_size*test) 
        
        input1_test = generate_input(H1_real,H1_image,SPC_test,N,batch_size*test,sigma)
        
        input2_test = generate_input(H2_real,H2_image,SPC_test,N,batch_size*test,sigma)
        
        hypo1= svm_user1.predict(input1_test)
    
        hypo2 = svm_user2.predict(input2_test)
        
        ERROR_user1[i,j] = 1-accuracy_score(test_label1, hypo1)
        
        ERROR_user2[i,j] = 1-accuracy_score(test_label2, hypo2)


error1 = np.mean((ERROR_user1),axis=1)
error2 = np.mean((ERROR_user2),axis=1)
    
plt.figure()
plt.semilogy(SNR_db,error1, ls = '--', marker = 'o',label='user1')
plt.semilogy(SNR_db,error2, ls = '--', marker = '+',label='user2')
plt.legend()
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('SER 2 SISO User in cluster')
plt.grid()
plt.savefig('result1')
plt.show()

















































