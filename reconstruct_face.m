function [reconstuction, reconst_no_mean] = reconstruct_face(eigenvectors, eigenvalues, mean_face, project_eigenfaces_train, k, type)

% Function performs face reconstruction

% Parameters:
%       eigenvalues: eigenvectors of the covariance matrix
%       eigenvectors: eigenvectors of the covariance matrix
%       mean_face: mean of 400 images of size 1x1024
%       project_eigenfaces_train: projected training img onto the subspace spanned by k-PC
%       k: number of proncipal components (eigenvectors to be used)
%       type: fndicate if to use eigendecomposition of T ('T') or
%       covariance_matrix ('S')

% Output:
%       reconstuction: outputs a recunstucted image (mean + u*projection)
%       reconst_no_mean: reconstruction before adding mean

if type == 'T'
% for the type = T only, we normalize the eigenvectors by the singular values

    eigenvectors = real(eigenvectors/sqrt(eigenvalues(1:k, 1:k))); 
  
    reconst_no_mean = eigenvectors * project_eigenfaces_train;
    reconstuction = mean_face' + reconst_no_mean;

else

    reconst_no_mean = eigenvectors * project_eigenfaces_train;
    reconstuction = mean_face' + reconst_no_mean;

end

end

