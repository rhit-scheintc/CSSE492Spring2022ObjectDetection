clear all
clc

%% Non-thm

%{
imgs = imageDatastore("ClassificationImages");
fileNames = imgs.Files;

idx = 1000;
for i = 1:size(fileNames, 1)
    file = cell2mat(fileNames(i));
    im = imread(file);
    im = im(150:end,:,:);
    if isempty(im) == 0
        imwrite(im, "C:\Users\scheintc\Documents\Robotics\Computer Vision\Underwater Video Challenge\VideoFrames\UnlabeledFrames\modified\Image" + (idx + (i - 1)) + ".jpg");
    end
end
%}

%% thm

imgs = imageDatastore("ClassificationImages");
fileNames = imgs.Files;

idx = 2000;
for i = 1:size(fileNames, 1)
    file = cell2mat(fileNames(i));
    im = imread(file);
    im = im(20:end,:,:);
    if isempty(im) == 0
        imwrite(im, "C:\Users\scheintc\Documents\Robotics\Computer Vision\Underwater Video Challenge\VideoFrames\UnlabeledFrames\modified\Image" + (idx + (i - 1)) + ".jpg");
    end
end