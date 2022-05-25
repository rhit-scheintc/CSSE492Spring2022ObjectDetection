clear all
clc

% Tutorial: https://www.geeksforgeeks.org/how-to-extract-frames-from-a-video-in-matlab/

load detectorEp160MBS32TS90

obj = VideoReader("LowDensity2.mp4");
writer = VideoWriter("LowDensity2Classified.mp4");
vid = read(obj);

frames = obj.NumberOfFrames;

open(writer)

for x = 1 : frames
    %vid(:,:,:,x);

    I = vid(:,:,:,x);

    [bboxes, scores, labels] = detect(yolov3Detector, I);

    if(isempty(labels) == 0)
        I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    end

    writeVideo(writer, I);

    (x/frames) * 100

end

close(writer);

%{

I = imread("UnlabeledImages\Image1011.jpg");
imtool(I)

[bboxes, scores, labels] = detect(yolov3Detector, I)

% Display the detections on image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);

figure
imshow(I)

%}
