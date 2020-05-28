function [SER_user1,SER_user2] = MIMO_NOMA_2USER_BPSK(n_iteration,n_symbol,N,M,SNR_db)

% N : number of transmit antenna
% M : number of recieve antenna
% n_iteration : number of iteration ( in one iteration channel matrix H1,H2 is fixed)
% n_symbol : length of data

symbol1 = 2 * randi([0,1],N,n_symbol,n_iteration) -1;

symbol2 = 2 * randi([0,1],N,n_symbol,n_iteration) -1;

%Generating channel matrix(Rayleigh fading channel), channel quality : user1 < user2  
H1 = (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

H2 = 2 * (randn(N,M,n_iteration)+1i*randn(N,M,n_iteration))/sqrt(2); 

%superposition coding, Power of SPC = 1, power allocation : user1 > user2

SPC = sqrt(0.8)*symbol1+ sqrt(0.2)*symbol2; 

%n_all=(randn(M,n_symbol,n_iteration)+1i*randn(M,n_symbol,n_iteration))/sqrt(2);

signal_power = mean(mean(mean(SPC.^2)));

ERROR_user1 = zeros(n_iteration,length(SNR_db));

ERROR_user2 = zeros(n_iteration,length(SNR_db));

for j = 1 : n_iteration       
    
        for i = 1: length(SNR_db)
        % 왜 sqrt(N)으로 나누는가? > 나누지 않으면 전체 시스템의 전송 파워가 N배 가 되어버림
        %즉 한 안테나에서 나오는 파워가 1이기 나누지 않는다면 POWER가 4인 SISO 시스템과 같아짐.
        %그렇게 되면 POWER GAIN인지 ANTENNA GAIN인지 판독 불가. 공정한 비교 X
        sigma=sqrt(signal_power*sqrt(N)/10^(SNR_db(i)/10));
        
        H_1 = H1(:,:,j);
        H_2 = H2(:,:,j);
        SPC_trans = SPC(:,:,j);
        noise = (randn(M,n_symbol)+1i*randn(M,n_symbol))/sqrt(2);
        %decoding user 1 symbol 
        
        y1= H_1*SPC_trans+ sigma*noise;  
        
        z1=(H_1'*H_1+(sigma^2)*eye(N))\H_1'*y1;
               
        x_decod = sign(real(z1));
              
        %n_errors1(i)=n_errors1(i)+sum(symbol1(:,j)~=x_decod);
        
        % decoding user2
        
        y2= H_2*SPC_trans+ sigma*noise;         
        
        % decoding user1's symbol in user2
        z2=(H_2' *H_2+(sigma^2)*eye(N))\H_2'*y2;
        
        x_decod_user1in2=sign(real(z2));
             
        %y3= awgn(H_2* (SPC(:,j)-0.8*x_decod_user1in2),SNR_db(i),'measured');
        y3= y2 - sqrt(0.8).*H_2*x_decod_user1in2;
        
        z3=(H_2' *H_2+(sigma^2)*eye(N))\H_2'*y3;
        
        x_decod_user2 = sign(real(z3));
        
        
        %n_errors2(i)=n_errors2(i)+sum(symbol2(:,j)~=x_decod_user2);
        
        ERROR_user1(j,i) = sum(abs((symbol1(:,:,j)-x_decod)/2),'all')/(n_symbol*N);
        
        ERROR_user2(j,i) = sum(abs((symbol2(:,:,j)-x_decod_user2)/2),'all')/(n_symbol*N);
        end
    
end

SER_user1 = mean(ERROR_user1);

SER_user2 = mean(ERROR_user2);

    
end


