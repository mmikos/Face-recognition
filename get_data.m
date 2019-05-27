function [train_faces, train_class, test_faces, test_class, no_img_train, no_img_test] = get_data(no_training_set)

% Function takes the number of training images per person and loads the
% corresponding training set and the main set with all faces. 
% Images used for training and testing are estracted from the main dataset
% using the inexes contained in the dataset. Function also performs
% normalization and returns the number of images extracted.

% Parameters:
%           no_training_set: number of training images per person


% Output:
%       train_faces: set of training images 
%       train_class: classes of training images
%       test_faces: set of test images 
%       test_class: classes of test images
%       no_img_test: number of images in the test set
%       no_img_train: number of images in the train set


    file_name = sprintf('faces/%dTrain/%d.mat', no_training_set, no_training_set);
    
    main = load('faces/ORL_32x32');
    set = load(file_name);

    train_faces = main.fea(set.trainIdx, :);
    train_class = main.gnd(set.trainIdx, :);
           
    test_faces = main.fea(set.testIdx, :);
    test_class = main.gnd(set.testIdx, :);
    
    % nomalize images
    train_faces = double(train_faces);
    train_faces = train_faces/255;   
    
    test_faces = double(test_faces);
    test_faces = test_faces/255;
    
    [no_img_test, ~] = size(test_faces);
    
    [no_img_train, ~] = size(train_faces);
    
end

