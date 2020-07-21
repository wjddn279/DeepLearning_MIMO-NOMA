function [x_decod] = ML(y1,hy1,modulation_order,N,n_symbol,k1)
m1 = repelem(y1,1,length(hy1)) - repmat(hy1,1,n_symbol);
m2 = ones(1,N) * abs(m1);
m3 = reshape(m2,[modulation_order^(2*N),n_symbol]);
[m4,m5] = min(m3,[],1);
x_decod = k1(:,m5);

end

