function [train_faces, train_class, test_faces, test_class] = get_data(no_training_img)

    file_name = sprintf('faces/%dTrain/%d.mat', no_training_img, no_training_img);
    
    load('faces/ORL_32x32');
    load(file_name);

    train_faces = fea(trainIdx, :);
    train_class = gnd(trainIdx, :);
           
    test_faces = fea(testIdx, :);
    test_class = gnd(testIdx, :);
    
    train_faces = double(train_faces);
    train_faces = train_faces/255;   
    
    test_faces = double(test_faces);
    test_faces = test_faces/255;
    
end

