clear
clc

load LabeledData

dataSize = size(gTruth.DataSource.Source, 1);

shuffledIndices = randperm(dataSize);
idx = floor(0.9 * length(shuffledIndices));

trainingIdxs = shuffledIndices(1:idx);
testingIdxs = shuffledIndices(idx+1:end);

% Splitting Images
trainImgs = gTruth.DataSource.Source(trainingIdxs);
testImgs = gTruth.DataSource.Source(testingIdxs);

% Splitting Labels
trainLabels = gTruth.LabelData(trainingIdxs, :);
testLabels = gTruth.LabelData(testingIdxs, :);

% Storing Images in Datastore
trainDS = imageDatastore(trainImgs);
testDS = imageDatastore(testImgs);

% Storing Labels in Datastore
trainBoxDS = boxLabelDatastore(trainLabels);
testBoxDS = boxLabelDatastore(testLabels);

% Combining data
trainingData = combine(trainDS, trainBoxDS);
testingData = combine(testDS, testBoxDS);

augmentedTrainingData = transform(trainingData, @augmentData);

networkInputSize = [227 227 3];
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

baseNetwork = squeezenet;
classNames = {'Creature'};

yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'});

preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(yolov3Detector, data));
preprocessedTestingData = transform(testingData, @(data)preprocess(yolov3Detector, data));

data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
%figure
%imshow(annotatedImage)

reset(preprocessedTrainingData);

numEpochs = 160;
miniBatchSize = 32;
learningRate = 0.01;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);

% Create subplots for the learning rate and mini-batch loss.
fig = figure;
[lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);



iteration = 0;
% Custom training loop.
for epoch = 1:numEpochs
          
    reset(mbqTrain);
    shuffle(mbqTrain);
        
    while(hasdata(mbqTrain))
        iteration = iteration + 1;
           
        [XTrain, YTrain] = next(mbqTrain);
            
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
    
        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
    
        % Determine the current learning rate value.
        currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
    
        % Update the detector learnable parameters using the SGDM optimizer.
        [yolov3Detector.Learnables, velocity] = sgdmupdate(yolov3Detector.Learnables, gradients, velocity, currentLR);
    
        % Update the state parameters of dlnetwork.
        yolov3Detector.State = state;
              
        % Display progress.
        displayLossInfo(epoch, iteration, currentLR, lossInfo);  
                
        % Update training plot with new points.
        updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
    end        
end

data = read(testingData);
I = data{1};

[bboxes, scores, labels] = detect(yolov3Detector, I)

% Display the detections on image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);

figure
imshow(I)
