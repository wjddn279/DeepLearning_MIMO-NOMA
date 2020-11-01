[SER_user1,SER_user2] = MIMO_NOMA_2USER_BPSK(1000,100000,4,4,2:2:30);

close all
figure
semilogy(2:2:30,SER_user1,'bp-','LineWidth',2);
hold on
semilogy(2:2:30,SER_user2,'kd-','LineWidth',2);
axis([5 30 10^-5 0.5])
grid on
legend('User1','User2');
xlabel('Average SNR,dB');

ylabel('Symbol 
Error Rate');