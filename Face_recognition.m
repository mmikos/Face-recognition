clear
    load('faces/ORL_32x32')
    load('faces/3Train/3.mat')
    load('faces/5Train/5.mat')
    load('faces/7Train/7.mat')
    
    im_size = 32; 
    no_faces = 400;

    im_norm = double(fea);
    im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));
%     im_norm = im_norm(2,:);
%%
    [m, n] = size(im_norm);

    mx = mean(im_norm)';
    mx_reshaped = reshape(mx, 32, 32);
    imshow(mx_reshaped,'Initialmagnification','fit')
    title('Mean face')
    colormap gray
    
    % Substract mean
    B = (im_norm'-mx)';
  
    % Covariance matrix
%     S = cov(B);
    S = 1/n * B*B';
    
    [eigenvectors, eigenvalues] = eig(S);
    
    face_space_coordinates = (B' * eigenvectors');
    
    eigenfaces=[];
    
    for k=1:no_faces
        eigface  = face_space_coordinates(:, k);
        eigenfaces{k} = reshape(eigface,im_size,im_size);
    end
    
x=diag(eigenvalues);
[xc,xci]=sort(x,'descend');% largest eigenval    
z  = [ eigenfaces{xci(1)}  eigenfaces{xci(2)}   eigenfaces{xci(3)} ; eigenfaces{xci(4)}     eigenfaces{xci(5)}   eigenfaces{xci(6)}];
figure(5),imagesc(z);;title('eigenfaces');colormap gray

%%
    
    rec = face_space_coordinates' * eigenvectors;

%     reconstructed = (mx + sum(face_space_coordinates))';
    
    reconsturcted_faces = (mx + rec')';
    
    face1 = reconsturcted_faces(100,:);
    face1_reshaped = reshape(face1, 32, 32);
%     imagesc(face1_reshaped)
    imshow(face1_reshaped,'Initialmagnification','fit')
    
    img_sum = zeros(im_size);

    for i = 1:no_faces
        
        im_reconstruct = reconsturcted_faces(i, :);
        im_reshaped = reshape(im_reconstruct, 32, 32);
        eigenfaces{i} = im_reshaped;
%         img_sum = img_sum + (st.img_extracted{i});
%         img_sum = img_sum + (st.img_extracted{k}/no_faces);
    end 

z  = [eigenfaces{1} eigenfaces{2} eigenfaces{3}; eigenfaces{4} eigenfaces{5} eigenfaces{6}];
figure(5),imshow(z,'InitialMagnification', 2000);
title('Reconstructed faces');
% eigenfaces=[];
% for i=1:no_faces
%     c  = W(:,i);
%     eigenfaces{i} = reshape(c, im_size, im_size);
% end
% 
% x=diag(D);
% [xc,xci]=sort(x,'descend');% largest eigenval
% z  = [eigenfaces{xci(1)}  eigenfaces{xci(2)}   eigenfaces{xci(3)} ; eigenfaces{xci(4)}     eigenfaces{xci(5)}   eigenfaces{xci(6)}];
% figure(5),imshow(z,'Initialmagnification','fit');

% X1=U*pinv(U*B+mx);
% X2=U(:,2)*pinv(U(:,2))*B+mx;
i = 2;
W = zeros(32, 32);
W(:, i) = (eigenvectors(:, i))' * B;

W(i, i)= reshape(W(i, i), 32, 32);

X_hat = mx + eigenvectors * W;



% plot(X1(1,:),X1(2,:),'.')
% hold on
% plot(X2(1,:),X2(2,:),'.')
% grid on 
% axis equal

PC1=(eigenvectors(:,[1,2])*pinv(eigenvectors(:,[1,2]))*B);
plot(PC1(1,:))
hold on
plot(PC1(2,:))

imshow(Y)


% 1/(n-1)* SS'
% M = mean(fea)

