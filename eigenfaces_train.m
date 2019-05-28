function [mean_face, eigenvectors, project_eigenfaces_train] = eigenfaces_train(train_faces, k, type)

% Function finds the meanface, covariance matrix and eigenvectors. An
% dimensionality was also implemented in order to make the
% eigendecomposition smaller in dimension by using T instead of 
% covariance_matrix - sigma.

% Parameters:
%       training_image: images from teh training set
%       k: number of proncipal components (eigenvectors to be used)
%       type: fndicate if to use eigendecomposition of T ('T') or
%       covariance_matrix ('S')
% Output:
%       mean_face: mean of 400 images of size 1x1024
%       eigenvectors: eigenvectors of the covariance matrix
%       project_eigenfaces_train: projected training img onto the subspace spanned by k-PC
    
[no_faces, ~] = size(train_faces);

mean_face = mean(train_faces);

img_subst_mean = (train_faces-mean_face)'; 

if type == 'S'
    
    covariance_matrix = (1/no_faces) * img_subst_mean * (img_subst_mean');

    [eigenvectors, eigenvalues] = eig(covariance_matrix); 

% applying dimensionality redution on eigen decomposition (T instead of
% covariance_matrix - sigma)

elseif type == 'T' 
    
    T = (1/no_faces) * (img_subst_mean') * (img_subst_mean);
    
    [eigenvectorsT, eigenvalues] = eig(T);
    
    eigenvectors = img_subst_mean * eigenvectorsT; 
    
    % eigenvectors of img_subst_mean are left singular vectors of 
    % img_subst_mean, hence dividing by sqrt(eigenvalues)

    eigenvectors = eigenvectors/(sqrt(eigenvalues));
    
    % we want unit vectors, hence we divide it by its lenght
    
    eigenvectors = eigenvectors/norm(eigenvectors);
    
else
    
    disp('Wrong type')

end

eigenvectors = eigenvectors(:, 1:k);

%Project training img onto the subspace spanned by k-PC
project_eigenfaces_train = eigenvectors' * (train_faces-mean_face)';

end

