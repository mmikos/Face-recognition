%% Original faces

clear

% extract original images from training set

m = 7; % no of training images per person (m = 3 for 3Train etc.)

im_size = 32; 

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(m);

original_images = [];

for j = 1:no_img_train
    org_face  = train_faces(j, :);
    original_images{j} = reshape(org_face,im_size,im_size);
end

figure(1); % Plot first 10 people (10*m faces)

for j = 1:10*m
    subplot(10, m, j);
    imagesc(original_images{j});
    colormap gray
    axis off
end

    
%% Eigenfaces and mean face

k = 70; % define how many eigenvecotrs will be used (PC)

type = 'S';

[mean_face, eigenvectors, ~] = eigenfaces_train(train_faces, k, type);

% plot meanface

mean_face_img = reshape(mean_face, 32, 32);

figure(2); %plot menaface
imshow(mean_face_img,'Initialmagnification','fit')
title('Mean face')

% extract eigenfaces from training set using k eigenvectors

k_eigenvectors=[];
    
 for j = 1:k
    eigface  = real(eigenvectors(:, j));
    k_eigenvectors{j} = reshape(eigface,im_size,im_size);
 end

figure(3); %plot k-eigenfaces

for j = 1:10*m
  subplot(10, m, j);
  imagesc(k_eigenvectors{j});
  colormap gray
  axis off
end

%% MAIN: Train + Test + Accuracy

clear

no_training_set = [3 5 7];

im_size = 32; 

k = 60; % no eigenvectors

accuracy = zeros(1, k);

reconstruction_error = zeros(1, k);

type = 'S';

tic()

for m = 1:length(no_training_set)

[train_faces, train_class, test_faces, test_class, ~, no_img_test] = get_data(no_training_set(m));

    for i = 1:k

        [mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, i, type);     
        
        classes_assigned = zeros(1, length(test_class));
        
        for j = 1:no_img_test
            
            class = eigenfaces_test(eigenvectors, mean_face, test_faces, train_class, j, project_eigenfaces_train);

            classes_assigned(j) = class;
        end
        
        [reconstuction, ~] = reconstruct_face(eigenvectors, mean_face, project_eigenfaces_train);
        
        accuracy(i) = sum(classes_assigned == test_class')/length(test_class)*100;

        reconstruction_error(i) = mean(mean((train_faces - reconstuction').^2));   
    end
    
    st.accuracy{m} = accuracy;
    st.reconstruction_error{m} = reconstruction_error;
    
    [val, arg] = max(accuracy);
    st.k_value_best{m} = arg; % take the k that guarantees max accuracy
    
end

toc() 

% using dimensionality redution for eigenspace decomposition: 9 sec
% using sigma covariance matrix for eigenspace decomposition: 149 sec
%% Plotting classification and reconstruction accuracy
    
figure(4);

plot(st.accuracy{1});
hold on 
plot(st.accuracy{2}) ; 
hold on 
plot(st.accuracy{3});
legend('3Train','5Train','7Train')
xlabel('Number of eigenvectors, k')
ylabel('Classification accuracy')
hold off

figure(5);

plot(st.reconstruction_error{1});
hold on 
plot(st.reconstruction_error{2}) ; 
hold on 
plot(st.reconstruction_error{3});
legend('3Train','5Train','7Train')
xlabel('Number of eigenvectors, k')
ylabel('Reconstruction error')
hold off

%% Reconstruction - extract images

% cheking reconstructions for the best k
type = 'T';

no_training_set = [3 5 7];

m = 2; % number of a positon in a vector no_training_set
<<<<<<< HEAD

n = no_training_set(m);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> cdec3d3... ff

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(n);

=======

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(n);

[mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);

=======
<<<<<<< HEAD

n = no_training_set(m);
<<<<<<< HEAD
=======
>>>>>>> cdec3d3... ff

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(n);

>>>>>>> origin/master
<<<<<<< HEAD
[mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);
=======
[mean_face, eigenvectors, eigenvalues, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);
>>>>>>> cdec3d3... ff
=======
<<<<<<< HEAD

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(n);

[mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);
>>>>>>> 240fc6d... h
=======
>>>>>>> origin/master
=======
=======
>>>>>>> cdec3d3... ff
>>>>>>> 11587ff... PCA FA

[train_faces, ~, ~, ~, no_img_train, ~] = get_data(n);

<<<<<<< HEAD
[mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);
<<<<<<< HEAD
>>>>>>> 240fc6d... h
=======
=======
[mean_face, eigenvectors, eigenvalues, project_eigenfaces_train] = eigenfaces_train(train_faces, st.k_value_best{m}, type);
>>>>>>> cdec3d3... ff
>>>>>>> 11587ff... PCA FA

>>>>>>> de327d9a324774ff67cefa6227fa89afb00f40af
[reconstuction, reconst_no_mean] = reconstruct_face(eigenvectors, mean_face, project_eigenfaces_train);

% extract reconstructed images without mean from training set

reconstructed_no_mean = [];

for j = 1:no_img_train
   eigface  = reconst_no_mean(:, j);
   reconstructed_no_mean{j} = reshape(eigface,im_size,im_size);
end

% extract reconstructed images from training set

reconstructed_image = [];

for j = 1:no_img_train
   recface  = reconstuction(:, j);
   reconstructed_image{j} = reshape(recface,im_size,im_size);
end

%% Plot reconstructed images


 figure(6);

 for j = 1:10*n
     subplot(10, n, j);
     imagesc(reconstructed_no_mean{j});
     colormap gray
     axis off
 end

figure(7);  

for j = 1:10*n
    subplot(10, n, j);
    imagesc(reconstructed_image{j});
    colormap gray
    axis off
end

