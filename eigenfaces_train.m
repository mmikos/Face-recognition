function [mean_face, eigenvectors] = eigenfaces_train(training_image, k)

% Function finds the meanface, covariance matrix and eigenvectors 

% Parameters:
%           training_image: images from teh training set
%           k: number of proncipal components (eigenvectors to be used)
% Output:
%       mean_face: mean of 400 images of size 1x1024
%       eigenvectors: eigenvectors of the covariance matrix
    
[no_faces, ~] = size(training_image);

mean_face = mean(training_image);

img_subst_mean = (training_image-mean_face)'; 

covariance_matrix = 1/no_faces * img_subst_mean * (img_subst_mean)';

[eigenvectors, ~] = eig(covariance_matrix);

eigenvectors = eigenvectors(:, 1:k);

end

