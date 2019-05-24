    clear
    load('faces/ORL_32x32')
    
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
    B = (im_norm'-mx);
  
    % Covariance matrix
    S = cov(B);
%     S = 1/n * B*B';
    
    [eigenvectors, diageigenvalues] = eig(S);
    
%     eigenvectors = eigenvectors(:, 1:10);
     
    face_space_coordinates = (eigenvectors' * B')';
    
    eigenfaces=[];
    
    for k=1:no_faces
        eigface  = face_space_coordinates(k, :);
        eigenfaces{k} = reshape(eigface,im_size,im_size);
    end   

    eigenvalues = diag(diageigenvalues);
    [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance

    face_vector  = [eigenfaces{index(1)} eigenfaces{index(2)} eigenfaces{index(3)};
                    eigenfaces{index(4)} eigenfaces{index(5)} eigenfaces{index(6)};
                    eigenfaces{index(7)} eigenfaces{index(8)} eigenfaces{index(9)};
                    eigenfaces{index(10)} eigenfaces{index(11)} eigenfaces{index(12)}];

    figure;
    imagesc(face_vector);
    title('eigenfaces');
    colormap gray
    
    %% Training
    
    clear
    load('faces/ORL_32x32')
    load('faces/3Train/3.mat')
%     load('faces/5Train/5.mat')
%     load('faces/7Train/7.mat')
    im_size = 32; 
    no_faces_train = length(trainIdx);
    k = 50;
    
    train3 = fea(trainIdx, :);

    train3 = double(train3);
    train3 = train3/255;

    
    [mean_face, eigenvectors, eigenvalues] = eigenfaces(train3, k); 
    
    [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance

    face_space_coordinates = (eigenvectors' * (train3'-mean_face)');
    
    eigenfaces=[];
    
     for i = 1:k
        eigface  = face_space_coordinates(i, :);
        eigenfaces{i} = reshape(eigface,im_size,im_size);
     end
       
    [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance  
     
    face_vector  = [eigenfaces{index(1)} eigenfaces{index(2)} eigenfaces{index(3)};
                    eigenfaces{index(4)} eigenfaces{index(5)} eigenfaces{index(6)};
                    eigenfaces{index(7)} eigenfaces{index(8)} eigenfaces{index(9)};];

    figure;
    imagesc(face_vector);
    title('eigenfaces');
    colormap gray
    
    %% Test
    
     load('faces/ORL_32x32')
     load('faces/3Train/3.mat')
%     load('faces/5Train/5.mat')
%     load('faces/7Train/7.mat')
    im_size = 32; 
    no_faces_test = length(testIdx);
    test3 = fea(testIdx, :);
    
    test3 = double(test3);
    test3 = test3/255;
    
    img_subst_mean = (test3' - mean_face);
    
    eigenface_space_test  = (eigenvectors' * img_subst_mean');
    
    reconsturction = mean_face + eigenface_space'; 
    
    