function [reconstuction, reconst_no_mean] = reconstruct_face(eigenvectors, mean_face, project_eigenfaces_train)

% Function performs face reconstruction

% Parameters:
%       eigenvectors: eigenvectors of the covariance matrix
%       mean_face: mean of 400 images of size 1x1024
%       project_eigenfaces_train: projected training img onto the subspace spanned by k-PC

% Output:
%       reconstuction: outputs a recunstucted image (mean + u*projection)
%       reconst_no_mean: reconstruction before adding mean


    reconst_no_mean = eigenvectors * project_eigenfaces_train;
    reconstuction = mean_face' + reconst_no_mean;

end

