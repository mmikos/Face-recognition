function [class] = eigenfaces_test(eigenvectors, mean_face, test_faces, train_class, no_img_test, project_eigenfaces_train)

% Function assigns the class to the image base on the training

% Parameters:
%       mean_face: mean of 400 images of size 1x1024
%       eigenvectors: eigenvectors of the covariance matrix
%       test_faces: set of test images 
%       train_class: classes of training images
%       no_img_test: number of images in the test set
%       project_eigenfaces_train: projected training img onto the subspace spanned by k-PC

% Output:
%       class: class assigned to an image


% Project test img onto the subspace spanned by eigenfaces
project_eigenfaces_test = eigenvectors' * (test_faces(no_img_test,:)-mean_face)';

% calculate distance between the subspaces; take real parts because
% distance cannot be computed between imaginary components
distance = pdist2(real(project_eigenfaces_test'), real(project_eigenfaces_train'));

[~, neighbors] = sort(distance);

% Take closest neighbor (smallest distance 1st from sorted list) and assign
% the class from the train set

class = train_class(neighbors(1)); 

end

