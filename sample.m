clear all
load('faces/ORL_32x32');
M=400;N=32;
avImg=zeros(N);

%% compute mean
for k=1:M
        im = fea(k, :);
        im_norm = double(im);
        im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));
        im_reshaped = reshape(im_norm, N, N);

        st.img_extracted{k} = im_reshaped;
        avImg = avImg + (1/M)*st.img_extracted{k};
end
figure(2),imshow(avImg,'Initialmagnification','fit');title('average')
%% normalize (remove mean)
for k=1:M
    st.dataAvg{k}  = st.img_extracted{k} -avImg;
end
z  = [ st.dataAvg{1}  st.dataAvg{2}   st.dataAvg{5}  ; st.dataAvg{3}     st.dataAvg{4}  st.dataAvg{6}];
figure(3),imshow(z,'Initialmagnification','fit');;title('z average')
%% generate A = [ img1(:)  img2(:) ...  imgM(:) ];
A = zeros(N*N,M);% (N*N)*M   2500*4
for k=1:M
    A(:,k) = st.dataAvg{k}(:);
end
% covariance matrix small dimension (transposed)
C = A'*A;
figure(4),imagesc(C);title('covariance')
%% eigen vectros  in small dimension
[Veigvec, Deigval ]  = eig(C);% v M*M e M*M only diagonal 4 eigen values
% eigan face in large dimension  A*veigvec is eigen vector of Clarge
Vlarge = A*Veigvec;% 2500*M*M*M  =2500 *M
% reshape to eigen face
eigenfaces=[];
for k=1:M
    c  = Vlarge(:,k);
    eigenfaces{k} = reshape(c,N,N);
end
x=diag(Deigval);
[xc,xci]=sort(x,'descend');% largest eigenval
z  = [ eigenfaces{xci(1)}  eigenfaces{xci(2)}   eigenfaces{xci(3)} ; eigenfaces{xci(4)}     eigenfaces{xci(5)}   eigenfaces{xci(6)}];
figure(5),imshow(z,'Initialmagnification','fit');;title('eigenfaces')
%% weights
nsel=5% select  eigen faces
for mi=1:M  % image number
  for k=1:nsel   % eigen face for coeff number
    wi(mi,k) =   sum(A(:,mi).* eigenfaces{xci(k)}(:)) ;
  end
end
%% classify new img  mic   
% folder work C:\Users\michaels.DSI\Desktop\faces\class\
testFaceMic = load('faces/3Train/3.mat');
testFaceMic  =rgb2gray(testFaceMic);
testFaceMic = imresize(testFaceMic,[N N]);
testFaceMic   =  im2single(testFaceMic);
%testFaceMic =  st.data{1}; test
figure(6), imshow(testFaceMic,'Initialmagnification','fit'); title('test face michael')
Aface = testFaceMic(:)-avImg(:); % normilized face
for(tt=1:nsel)
  wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
end
% compute distance
for mi=1:M  
    fsumcur=0;
    for(tt=1:nsel)
        fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
    end
    diffWeights(mi) =   sqrt( fsumcur);
end
% mic classified as 5 ..
%% classify new img  linoy
testFaceLinoy = imread('100_2120.jpg','jpg');
testFaceLinoy  =rgb2gray(testFaceLinoy);
testFaceLinoy = imresize(testFaceLinoy,[N N]);
testFaceLinoy   =  im2single(testFaceLinoy);
figure(7), imshow(testFaceLinoy,'Initialmagnification','fit'); title('test face linoy')
Aface = testFaceLinoy(:)-avImg(:);
for(tt=1:nsel)
  wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
end
% compute distance
for mi=1:M  
    fsumcur=0;
    for(tt=1:nsel)
        fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
    end
    diffWeights(mi) =   sqrt( fsumcur);
end
% linoy classified as libi
%% libi3.jpg
testFaceLibi = imread('libi3.jpg','jpg');
testFaceLibi  =rgb2gray(testFaceLibi);
testFaceLibi = imresize(testFaceLibi,[N N]);
testFaceLibi   =  im2single(testFaceLibi);
figure(8), imshow(testFaceLibi,'Initialmagnification','fit'); title('test face testFaceLibi')
Aface = testFaceLibi(:)-avImg(:);
wface=[];
for(tt=1:nsel)
  wface(tt)  =  sum(Aface.* eigenfaces{xci(tt)}(:)) ;
end
% compute distance
for mi=1:M  
    fsumcur=0;
    for(tt=1:nsel)
        fsumcur = fsumcur + (wface(tt) -wi(mi,tt)).^2;
    end
    diffWeights(mi) =   sqrt( fsumcur);
end
diffWeights  =diffWeights.';