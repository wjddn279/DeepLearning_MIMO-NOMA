

N = 4;
M = 4;
n_symbol =480000;
n_iteration = 30;
SNR_db = [0:2:20];
M2 = M/2;
modulation_order = 2;
pilot_len = 10;
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
%H1 = (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

%H2 = 2 * (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

%superposition coding, Power of SPC = 1, power allocation : user1 > user2

SPC = sqrt(0.8)*symbol1+ sqrt(0.2)*symbol2; 

n_all=(randn(M,n_symbol,n_iteration)+1i*randn(M,n_symbol,n_iteration))/sqrt(2);

signal_power = mean(mean(mean(SPC.^2)));

ERROR_user1 = zeros(n_iteration,length(SNR_db));

ERROR_user2 = zeros(n_iteration,length(SNR_db));

ERROR_user1_est_1 = zeros(n_iteration,length(SNR_db));

ERROR_user2_est_2 = zeros(n_iteration,length(SNR_db));

x_decod = zeros(M,n_iteration);

x_decod_user2 = zeros(M,n_iteration);


like1 = zeros(1,modulation_order^N);

like2 = zeros(1,modulation_order^N);


for j = 1 : n_iteration       
        
        j    
        H_1 = H1(:,:,j);
        H_2 = H2(:,:,j);
    
        for i = 1: length(SNR_db)

            sigma=sqrt(signal_power*sqrt(N)/(10^(SNR_db(i)/10)));
            SPC_trans = SPC(:,:,j);
            noise = n_all(:,:,j);
            %decoding user 1 symbol
        
            y1= H_1*SPC_trans+ sigma*noise;
            
            hy1 = H_1*point;
            
            y2= H_2*SPC_trans+ sigma*noise;      

            hy2 = H_2*point;
            
            S1 = repmat(eye(N),1,pilot_len)';
            S2 = repmat(eye(N),1,pilot_len)';
            y1_pilot = S1*H_1+sigma*(randn(N*pilot_len,N)+1i*randn(N*pilot_len,N))/sqrt(2);
            y2_pilot = S2*H_2+sigma*(randn(N*pilot_len,N)+1i*randn(N*pilot_len,N))/sqrt(2);
            
            H1_est = (S1'*S1)\S1'*y1_pilot;
            H2_est = (S2'*S2)\S2'*y2_pilot;
            
            hy1_est = H1_est * point;
            hy2_est = H2_est * point;
                       
            x_decod_est = ML(y1,hy1_est,modulation_order,N,n_symbol,k1);
            x_decod_user2_est = ML(y2,hy2_est,modulation_order,N,n_symbol,k2);
            
            ERROR_user1_est_1(j,i) = sum(abs((symbol1(:,:,j)-x_decod_est)/2),'all')/(n_symbol*N);
            ERROR_user2_est_2(j,i) = sum(abs((symbol2(:,:,j)-x_decod_user2_est)/2),'all')/(n_symbol*N);
        end
    
end

SER_user1_ce = mean(ERROR_user1_est_1);

SER_user2_ce = mean(ERROR_user2_est_2);


close all
figure
semilogy(SNR_db,SER_user1_est,'ro-','LineWidth',2);
hold on
semilogy(SNR_db,SER_user2_est,'y+-','LineWidth',2);
grid on
legend('User1.ce', 'User2.ce');
xlabel('Average Eb/No,dB');
ylabel('Bit Error Rate');
title('BER for BPSK modulation with 2x2 MIMO and MMSE equalizer (Rayleigh channel)');