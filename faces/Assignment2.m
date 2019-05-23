
load('ORL_32x32')
load('3Train/3.mat')

im = fea(2, :);
im_norm = double(im);
im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));

im_reshaped = reshape(im_norm, 32, 32);

imshow(im_reshaped)
% colormap gray



%%
[m, n] = size(im_reshaped);
mx=mean(im_reshaped')';
B = im_reshaped-mx*ones(1,n);


%Covariance matrix
S = 1/(n-1) * B*B';

[U, D] = eig(S);

% D=fliplr(flipud(D));
% U=fliplr(U);

Y = U'* B;

X1=U*pinv(U*B+mx*ones(1,n));
X2=U(:,2)*pinv(U(:,2))*B+mx*ones(1,n);

% plot(X1(1,:),X1(2,:),'.')
% hold on
% plot(X2(1,:),X2(2,:),'.')
% grid on 
% axis equal

PC1=(U(:,[1,2])*pinv(U(:,[1,2]))*B);
plot(PC1(1,:))
hold on
plot(PC1(2,:))

imshow(Y)


% 1/(n-1)* SS'
% M = mean(fea)

