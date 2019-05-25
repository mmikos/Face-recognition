function [mean_face, eigenvectors] = eigenfaces_train(training_image, k)
    
    [no_faces, face_size] = size(training_image);
    
    mean_face = mean(training_image);
    
    img_subst_mean = (training_image-mean_face)'; 
    
%     S = cov(img_subst_mean);
    S = 1/no_faces * img_subst_mean * img_subst_mean';
    
    [eigenvectors, diagmatrix] = eig(S);
    
    eigenvectors = eigenvectors(:, 1:k);
    
    eigenvalues = diag(diagmatrix(1:k, 1:k));
    
end

