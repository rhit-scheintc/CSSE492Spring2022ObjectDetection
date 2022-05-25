clear all
clc

imgs = imageDatastore('C:\Users\scheintc\Documents\Robotics\Computer Vision\Underwater Video Challenge\ClassOrganization\ClassificationImages', 'FileExtensions', [".jpg"]);

numFiles = size(imgs.Files, 1);
numTrainingImages = 1000;

inds = floor(linspace(1, numFiles, numTrainingImages));

for i = 1:numTrainingImages
    
    imageFilePath = cell2mat(imgs.Files(inds(i)));
    im = imread(imageFilePath);

    if sum(imageFilePath(end-6:end) == 'thm.jpg') == 7
        im = im(20:end,:,:);
    else
        im = im(150:end,:,:);
    end

    imwrite(im, "UnlabeledImages\Image" + (i-1) + ".jpg");

end

%{
fileName = cell2mat(imgs.Files(1));
I = imread(fileName);
imshow(I(150:end,:,:))
%}