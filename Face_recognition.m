    clear
    load('faces/ORL_32x32')
    
    im_size = 32; 
    no_faces = 400;
    
    %normalize
    im = double(fea);
    im_norm = im/255;

    [no_faces, face_size] = size(im_norm);

    mean_face = mean(im_norm);
    mx_reshaped = reshape(mean_face, 32, 32);
    imshow(mx_reshaped,'Initialmagnification','fit')
    title('Mean face')
    
    % Substract mean
    img_subst_mean = (im_norm-mean_face)';
  
    % Covariance matrix
%     S2 = cov(B);
    S = 1/no_faces * img_subst_mean * img_subst_mean';
    
    [eigenvectors, diagmatrix] = eig(S);
    
    k = 63;
    
    k_eigenvectors=[];
    
     for j = 1:k
        eigvec  = eigenvectors(:, j);
        k_eigenvectors{j} = reshape(eigvec,im_size,im_size);
     end
     
    figure();
    for j = 1:k
      subplot(8,9,j);
      imagesc(k_eigenvectors{j});
      colormap gray
      axis off
    end
%     title('Top eigenvectors')
     
    eigenvectors = eigenvectors(:, 1:k);
    
    face_space_coordinates = (eigenvectors' * img_subst_mean);
    
    eigenfaces=[];
    
    for j = 1:no_faces
        eigface  = face_space_coordinates(j, :);
        eigenfaces{j} = reshape(eigface,im_size,im_size);
    end   

%     eigenvalues = diag(diageigenvalues);
%     [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance
    figure();
    for j = 1:k
      subplot(4,5,j);
      imagesc(eigenfaces{j});
      colormap gray
      axis off
    end
    
    
    %% Training
    
%     clear
    load('faces/ORL_32x32')
    load('faces/7Train/7.mat')

    train_faces = fea(trainIdx, :);
    train_class = gnd(trainIdx, :);
    
    [no_faces_train, face_size_train] = size(train_faces);
    im_size = 32; 
    
    train_faces = double(train_faces);
    train_faces = train_faces/255;     

    k = 10;
        
    [mean_face, eigenvectors] = eigenfaces_train(train_faces, k); 
    
    k_eigenvectors=[];
    
     for j = 1:k
        eigface  = eigenvectors(:, j);
        k_eigenvectors{j} = reshape(eigface,im_size,im_size);
     end
     
    figure();
    for j = 1:k
      subplot(4,5,j);
      imagesc(k_eigenvectors{j});
      colormap gray
      axis off
    end
    
%     title('Top eigenvectors')
%     [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance

    project_eigenfaces_train = (eigenvectors' * (train_faces-mean_face)');
    
    %% Test

    test_faces = fea(testIdx, :);
    test_class = gnd(testIdx, :);
    
    [no_faces_test, face_size_test] = size(test_faces);
    im_size = 32; 
    
    test_faces = double(test_faces);
    test_faces = test_faces/255;
             
%     [sorted_values, index] = sort(eigenvalues,'descend');% largest eigenvalue first - biggest variance
%     class = zeros(1, length(test3_class));
%% Test + Accuracy
    
%     clear
    
    load('faces/ORL_32x32')
    load('faces/5Train/5.mat')

    train_faces = fea(trainIdx, :);
    train_class = gnd(trainIdx, :);
           
    test_faces = fea(testIdx, :);
    test_class = gnd(testIdx, :);
    
    train_faces = double(train_faces);
    train_faces = train_faces/255;   
    
    test_faces = double(test_faces);
    test_faces = test_faces/255;
    
    [no_faces_test, face_size_test] = size(test_faces);
    
    [no_faces_train, face_size_train] = size(train_faces);
    
    k = 100;
    
    accuracy5 = zeros(1, length(k));
    
    for i = 1:k
        
        [mean_face, eigenvectors] = eigenfaces_train(train_faces, i); 
        
        project_eigenfaces_train = (eigenvectors' * (train_faces-mean_face)');
        
        class = zeros(1, length(test_class));
        
        for j = 1:no_faces_test

            project_eigenfaces_test = (eigenvectors' * (test_faces(j,:)-mean_face)');

            distance = pdist2(project_eigenfaces_test', project_eigenfaces_train');

            [sorted_values, neighbors] = sort(distance);
            
            % taking the closest neighbor (first from sorted list) and 
            % assigning the class from the train set
            
            class(j) = train_class(neighbors(1)); 
            
        end
        
        accuracy5(i) = sum(class == test_class')/length(test_class)*100;
                  
    end   
    
    [val, arg] = max(accuracy5);
    k_value_best = arg;

    %% Plotting
    figure();
    
    plot(accuracy3);

    hold on 
    plot(accuracy5) ; 
    hold on 
    plot(accuracy7);
    legend('3Train','5Train','7Train')
    xlabel('Number of eigenvectors, k')
    ylabel('Classification accuracy')
    hold off
%     reconsturction = mean_face + eigenface_space'; 

%     d = median(train3_class(neighbors,1:k));

%% Reconstruction

[mean_face, eigenvectors] = eigenfaces_train(train_faces, k_value_best); 
        
project_eigenfaces_train = (eigenvectors' * (train_faces-mean_face)');

reconstuction = mean_face' + eigenvectors * project_eigenfaces_train;

    
    reconstructed_image = [];
    
     for j = 1:no_faces_train
        eigface  = reconstuction(:, j);
        reconstructed_image{j} = reshape(eigface,im_size,im_size);
     end
     
    figure(1);
    for j = 1:no_faces_train/4
      subplot(10,5,j);
      imagesc(reconstructed_image{j});
      colormap gray
      axis off
    end
    
       original_images = [];
    
     for j = 1:no_faces_train
        org_face  = train_faces(j, :);
        original_images{j} = reshape(org_face,im_size,im_size);
     end
     
    figure(2);
    for j = 1:no_faces_train/4
      subplot(10,5,j);
      imagesc(original_images{j});
      colormap gray
      axis off
    end