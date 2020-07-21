clc; clear;

N = 4;
M = 4;
n_symbol = 200000;
n_iteration = 1;
SNR_db = [2:2:10];
M2 = M/2;
modulation_order = 2;
d = zeros(2^M,M);
for i = 1:4
    a = [[0,0];[0,1];[1,0];[1,1]];
    b1 = repmat(a(i,:),M);
    b = reshape(b1(:,1:2),[M,M2]);
    c = [b,a];
    d(4*i-3:4*i,:) = c;
end
point = d';

point = 2*point - 1;

k2 = repmat(point,modulation_order^N);
k2 = k2(1:4,:);

k1 = zeros(N,modulation_order^(N*2));
for i = 1 : modulation_order^N
    p = repmat(point(:,i),modulation_order^N);
    k1(:,16*i-15:16*i) =  p(1:N,:);
end

point = sqrt(0.8)*k1 + sqrt(0.2)*k2;


        
symbol1 = 2 * randi([0,1],N,n_symbol,n_iteration) -1;

symbol2 = 2 * randi([0,1],N,n_symbol,n_iteration) -1;

%Generating channel matrix(Rayleigh fading channel), channel quality : user1 < user2  
H1 = (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

H2 = 2 * (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

%superposition coding, Power of SPC = 1, power allocation : user1 > user2

SPC = sqrt(0.8)*symbol1+ sqrt(0.2)*symbol2; 

n_all=(randn(M,n_symbol,n_iteration)+1i*randn(M,n_symbol,n_iteration))/sqrt(2);

signal_power = mean(mean(mean(SPC.^2)));

ERROR_user1 = zeros(n_iteration,length(SNR_db));

ERROR_user2 = zeros(n_iteration,length(SNR_db));

x_decod = zeros(M,n_iteration);

x_decod_user1in2 = zeros(M,n_iteration);

x_decod_user2 = zeros(M,n_iteration);

like1 = zeros(1,modulation_order^N);

like2 = zeros(1,modulation_order^N);

%H_1 = (randn(N,M)+1i*randn(N,M))/sqrt(2);
%H_2 =  2 * (randn(N,M)+1i*randn(N,M))/sqrt(2);

H1_1 = [[0.7565  ,  0.0246 ,   0.4708 ,  -0.6736];
   [-1.3732 ,   0.7148   , 0.5825 ,   0.7321];
  [ -0.8394  , -1.3591  , -0.0328  ,  0.4201];
   [ 0.4125   , 0.3928 ,   0.3829   ,-0.9459]]+1i*[[   -0.9027  ,  0.3607 ,   1.7073 ,   1.4104];
    [0.2397  ,  1.2623  , -0.0489 ,   0.7416];
    [0.6492   ,-1.5532  , -0.0702  , -2.4573];
    [1.3094   , 0.5485 ,   1.0087  , -0.2718]];

H2_1 = [[    2.0915 ,  -0.0277 ,  -2.8472 ,   0.3049];
   [-1.2719  , -0.9617  , -0.8682  ,  1.3288];
    [1.1487  ,  0.9615  , -0.5566  , -2.5416];
    [1.5289  , -3.2012 ,  -0.3904  ,  1.0534]] + 1i*[[    2.9684  , -0.1825 ,  -0.3127 ,   1.0636];
   [-0.2159 ,   0.7222  ,  1.0957 ,  -0.9675];
   [-1.6677  , -1.0497  ,  0.1018  , -0.0674];
    [0.7354  ,  0.0341 ,   1.8606  , -0.1552]];

for j = 1 : n_iteration       
        
        j       
        H_1 = H1_1;
        H_2 = H2_1;
    
        for i = 1: length(SNR_db)

            sigma=sqrt(signal_power/(sqrt(N)*10^(SNR_db(i)/10)));
            SPC_trans = SPC(:,:,j);
            noise = n_all(:,:,j);
            %decoding user 1 symbol 
        
            y1= H_1*SPC_trans+ sigma*noise;  
        
            hy1 = H_1*point;
        
            for it = 1 : length(y1)
            
                for l = 1 :length(hy1)
                    like1(l) = norm(y1(:,it)-hy1(:,l));
                end
                [q,lo] = min(like1);
                x_decod(:,it) = k1(:,lo);
            end
        
              
            %n_errors1(i)=n_errors1(i)+sum(symbol1(:,j)~=x_decod);
        
            % decoding user2

            y2= H_2*SPC_trans+ sigma*noise;      

            hy2 = H_2*point;

            % decoding user1's symbol in user2
            for it = 1 : length(y1)

                for l = 1 :length(hy1)
                    like2(l) = norm(y2(:,it)-hy2(:,l));
                end
                [q,lo] = min(like2);
                x_decod_user2(:,it) = k2(:,lo);
            end


            %n_errors2(i)=n_errors2(i)+sum(symbol2(:,j)~=x_decod_user2);

            ERROR_user1(j,i) = sum(abs((symbol1(:,:,j)-x_decod)/2),'all')/(n_symbol*N);

            ERROR_user2(j,i) = sum(abs((symbol2(:,:,j)-x_decod_user2)/2),'all')/(n_symbol*N);
        end
    
end

SER_user1 = mean(ERROR_user1);


SER_user2 = mean(ERROR_user2);

close all
figure
semilogy(SNR_db,ERROR_user1,'bp-','LineWidth',2);
hold on
semilogy(SNR_db,ERROR_user2,'kd-','LineWidth',2);
grid on
legend('User1', 'User2');
xlabel('Average Eb/No,dB');
ylabel('Bit Error Rate');
title('BER for BPSK modulation with 2x2 MIMO and MMSE equalizer (Rayleigh channel)');