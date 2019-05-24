function [mean_face, eigenvectors, eigenvalues] = eigenfaces(image, k)
    
%     im_norm = double(image);
%     im_norm = (im_norm-min(im_norm(:)))/(max(im_norm(:))-min(im_norm(:)));

    mean_face = mean(image)';
    
    img_subst_mean = (image'-mean_face);
    
    S = cov(img_subst_mean);
    
    [eigenvectors, diagmatrix] = eig(S);
    
    eigenvectors = eigenvectors(:, 1:k);
    
    eigenvalues = diag(diagmatrix(1:k, 1:k));
    
%     face_space_coordinates = (eigenvectors' * img_subst_mean');
%     
%     eigenfaces=[];
%     
%      for k=1:no_faces
%         eigface  = face_space_coordinates(k, :);
%         eigenfaces{k} = reshape(eigface,im_size,im_size);
%      end
%     
end

