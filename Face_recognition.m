clear
    load('faces/ORL_32x32')
    load('faces/3Train/3.mat')

    im_size = 32; 
    no_faces = 40;

%     im = fea(400, :);
    im_norm = double(fea);
%     im_norm = im_norm/256;
    im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));
%     
%     im_reshaped = reshape(im_norm, 32, 32);

%     img_sum = zeros(im_size);

%     for i = 1:no_faces
%         
%         im = fea(i, :);
%         im_norm = double(im);
%         im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));
%         im_reshaped = reshape(im_norm, 32, 32);
% 
%         st.img_extracted{i} = im_reshaped;
%         img_sum = img_sum + (st.img_extracted{i});
% %         img_sum = img_sum + (st.img_extracted{k}/no_faces);
%     end
%     
%     img_mean = img_sum/no_faces; %average face
% 
% 
%     for i = 1:no_faces
%         st.img_extracted_centered{i}  = st.img_extracted{i} - img_mean;
%     end
    



% imshow(im_reshaped)
% colormap gray

% figure;
% imshow(struct.img_extracted{i},'Initialmagnification','fit');



%%

% A = zeros(im_size*im_size,no_faces);% (N*N)*M   2500*4
% 
% for i=1:no_faces
%     A(:,i) =  struct.img_extracted_mean{i}(:);
% end
% % covariance matrix small dimension (transposed)
% S = A' * A;
% figure(4),imagesc(C);title('covariance')

    [m, n] = size(im_norm);

    mx = mean(im_norm)';
    mx_reshaped = reshape(mx, 32, 32);
%     imshow(mx_reshaped)
    imagesc(mx_reshaped)
    title('Mean face')
    colormap gray
    
    B = im_norm'-mx;
    k = 10;
    %Covariance matrix
    S = 1/n * B*B';
%     S2 = cov(B');

    [eigenvectors, eigenvalues] = eig(S);
    
    eigen = eigenvectors(:, 1000);
    eigen_reshaped = reshape(eigen, 32, 32);
%     imshow(eigen_reshaped,'InitialMagnification', 2000)
    imagesc(eigen_reshaped)
    title('Eigen vector 1000')
    colormap gray
% D=fliplr(flipud(D));
% U=fliplr(U);

    face_space_coordinates = (eigenvectors' * B)';
%     imshow(eigen_reshaped,'InitialMagnification', 2000)
    imagesc(eigen_reshaped)
    title('Eigen vector 1000')
    colormap gray


    reconsturcted_faces = (mx + (face_space_coordinates * eigenvectors)')';
    
    face1 = reconsturcted_faces(1,:);
    face1_reshaped = reshape(face1, 32, 32);
    imshow(face1_reshaped,'InitialMagnification', 2000)
    
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
