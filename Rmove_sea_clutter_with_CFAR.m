dataURL = 'https://ssd.mathworks.com/supportfiles/radar/data/MaritimeRadarPPI.zip';
unzip(dataURL);
imdata = load('MaritimeRadarPPI.mat');
imgs = zeros(626, 626, 1, 84, 'single');
resps = zeros(626, 626, 1, 84, 'single');

for ind = 1:84
    imgs(:,:,1,ind) = imdata.(sprintf('img%d',ind));
    resps(:,:,1,ind) = imdata.(sprintf('resp%d',ind));
end
clearvars imdata

layers = imageInputLayer([626 626]);
layers(end+1) = convolution2dLayer([5 5], 1, 'NumChannels', 1, 'Padding', 'same');
layers(end+1) = batchNormalizationLayer;
layers(end+1) = leakyReluLayer(0.2);
layers(end+1) = convolution2dLayer([6 6], 4, 'NumChannels', 1, 'Padding', 'same');
layers(end+1) = batchNormalizationLayer;
layers(end+1) = leakyReluLayer(0.2);
layers(end+1) = convolution2dLayer([5 5], 1, 'NumChannels', 4, 'Padding', 'same');
layers(end+1) = batchNormalizationLayer;
layers(end+1) = leakyReluLayer(0.2);
layers(end+1) = regressionLayer;

rng default
trainSet = 1:60;
valSet = 61:70;

opts = trainingOptions("adam", ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 20, ...
    'Shuffle', "every-epoch", ...
    'InitialLearnRate', 0.1, ...
    'ValidationData', {imgs(:,:,:,valSet), resps(:,:,:,valSet)}, ...
    'ValidationFrequency', 25, ...
    'Verbose', true);

if true
    [net, info] = trainNetwork(imgs(:,:,:,trainSet), resps(:,:,:,trainSet), layers, opts);
end

evalSet = 71:84;

% Apply clutter removal and evaluate
detectionProbBefore = zeros(numel(evalSet), 1);
detectionProbAfter = zeros(numel(evalSet), 1);
detectionProbAfterOSCFAR = zeros(numel(evalSet), 1);
SNRdB =1000* log10( zeros(numel(evalSet), 1));

for ind = 1:numel(evalSet)
    % Apply clutter removal to the input image
    img = imgs(:,:,1,evalSet(ind));
    resp_desired = resps(:,:,1,evalSet(ind));
    
    % Apply CA-CFAR
    clutterRemovedImg = applyCACFAR(img);
    
    % Apply OS-CFAR
    clutterRemovedImgOSCFAR = applyOSCFAR(img);
    
    % Normalize the clutter-removed images and desired response
    clutterRemovedImg = clutterRemovedImg / max(clutterRemovedImg(:));
    clutterRemovedImgOSCFAR = clutterRemovedImgOSCFAR / max(clutterRemovedImgOSCFAR(:));
    resp_desired = resp_desired / max(resp_desired(:));
    
    % Predict the response using the trained network
    if exist('net', 'var')
        resp_act = predict(net, clutterRemovedImg);
        resp_act(resp_act < 0) = 0;
        resp_act = resp_act / max(resp_act(:));
        
        resp_actOSCFAR = predict(net, clutterRemovedImgOSCFAR);
        resp_actOSCFAR(resp_actOSCFAR < 0) = 0;
        resp_actOSCFAR = resp_actOSCFAR / max(resp_actOSCFAR(:));
    else
        error("Neural network 'net' is not defined. Please train the network first or load a pre-trained network.")
    end
    
    % Calculate detection probability
    detectionProbBefore(ind) = calculateDetectionProb(img, resp_desired);
    detectionProbAfter(ind) = calculateDetectionProb(clutterRemovedImg, resp_desired);
    detectionProbAfterOSCFAR(ind) = calculateDetectionProb(clutterRemovedImgOSCFAR, resp_desired);
    
    % Calculate SNR in dB
    SNRdB(ind) = 1000 * log10(calculateSNR(img,resp_desired));
    
    % Plot original image, clutter-removed image, and predicted response
    plotImages(img, clutterRemovedImg, resp_act, clutterRemovedImgOSCFAR, resp_actOSCFAR);
end

% Plot detection probability versus SNR
plotDetectionVsSNR(detectionProbBefore, detectionProbAfter, detectionProbAfterOSCFAR, SNRdB);

function clutterRemovedImg = applyCACFAR(img)
    % Apply CA-CFAR algorithm to remove clutter from the input image
    % Define CA-CFAR parameters
    guardCells = [4 4];
    trainingCells = [10 10];
    falseAlarmRate = 1e-4;

    % Convert input image to linear scale
    img_linear = 10.^(img / 20);

    % Apply CA-CFAR
    clutterRemovedImg_linear = cfarDetection(img_linear, guardCells, trainingCells, falseAlarmRate);

    % Convert back to logarithmic scale
    clutterRemovedImg = 20 * log10(clutterRemovedImg_linear);
end

function clutterRemovedImgOSCFAR = applyOSCFAR(img)
    % Apply OS-CFAR algorithm to remove clutter from the input image
    % Define OS-CFAR parameters
    guardCells = [4 4];
    trainingCells = [10 10];
    falseAlarmRate = 1e-4;

    % Convert input image to linear scale
    img_linear = 10.^(img / 20);

    % Apply OS-CFAR
    clutterRemovedImg_linear = oscfarDetection(img_linear, guardCells, trainingCells, falseAlarmRate);

    % Convert back to logarithmic scale
    clutterRemovedImgOSCFAR = 20 * log10(clutterRemovedImg_linear);
end

% Define the remaining local functions (cfarDetection, oscfarDetection, calculateDetectionProb, calculateSNR, plotImages, plotDetectionVsSNR)

function clutterRemovedImg_linear = cfarDetection(img_linear, guardCells, trainingCells, falseAlarmRate)
    % Apply Constant False Alarm Rate (CFAR) detection

    % Get image dimensions
    [M, N] = size(img_linear);

    % Define window size
    windowSize = (2 * guardCells + 2 * trainingCells + 1);

    % Pad the image
    img_padded = padarray(img_linear, guardCells, 'replicate');

    % Create output image
    clutterRemovedImg_linear = zeros(M, N);

    % Slide the window across the image
    for i = 1:M
        for j = 1:N
            if i+windowSize(1)-1 <= M && j+windowSize(2)-1 <= N
                % Extract the local region
                localRegion = img_padded(i:i+windowSize(1)-1, j:j+windowSize(2)-1);

                % Calculate the threshold
                sortedVals = sort(localRegion(:), 'ascend');
                threshold = sortedVals(trainingCells(1) * trainingCells(2) + 1);

                % Compare the central pixel to the threshold
                if img_linear(i, j) > threshold
                    clutterRemovedImg_linear(i, j) = img_linear(i, j);
                end
            end
        end
    end
end

function clutterRemovedImg_linear = oscfarDetection(img_linear, guardCells, trainingCells, falseAlarmRate)
    % Apply Order Statistic CFAR (OS-CFAR) detection

    % Get image dimensions
    [M, N] = size(img_linear);

    % Define window size
    windowSize = (2 * guardCells + 2 * trainingCells + 1);

    % Pad the image
    img_padded = padarray(img_linear, guardCells, 'replicate');

    % Create output image
    clutterRemovedImg_linear = zeros(M, N);

    % Slide the window across the image
    for i = 1:M
        for j = 1:N
            if i+windowSize(1)-1 <= M && j+windowSize(2)-1 <= N
                % Extract the local region
                localRegion = img_padded(i:i+windowSize(1)-1, j:j+windowSize(2)-1);

                % Calculate the threshold
                sortedVals = sort(localRegion(:), 'ascend');
                threshold = sortedVals(trainingCells(1) * trainingCells(2) + 1);

                % Compare the central pixel to the threshold
                if img_linear(i, j) > threshold
                    clutterRemovedImg_linear(i, j) = img_linear(i, j);
                end
            end
        end
    end
end

function detectionProb = calculateDetectionProb(clutterRemovedImg, resp_desired)
    % Calculate detection probability

    % Set threshold for detection
    threshold = 0.5;

    % Threshold the clutter-removed image
    clutterRemovedImg_thresholded = clutterRemovedImg > threshold;

    % Calculate true positive rate
    truePositive = clutterRemovedImg_thresholded & (resp_desired > threshold);
    detectionProb = sum(truePositive(:)) / sum(resp_desired(:) > threshold);
end

function SNR = calculateSNR(img,resp_desired)
    % Calculate signal-to-noise ratio (SNR)

    % Calculate signal power
    signalPower = mean(resp_desired(:).^2);

    % Calculate noise power
    noisePower = mean((img(:) - resp_desired(:)).^2);

    % Calculate SNR
        SNR = signalPower / noisePower;
% Convert SNR to dB
%     SNRdB = 100 * log10(SNR);
end

function plotImages(img, clutterRemovedImg, resp_act, clutterRemovedImgOSCFAR, resp_actOSCFAR)
    % Plot original image, clutter-removed image, and predicted response

    fh = figure;
    subplot(2, 3, 1)
    imagesc(20 * log10(img / max(img(:))))
    colorbar
    axis equal
    axis tight
    title('Original Image')

    subplot(2, 3, 2)
    imagesc(clutterRemovedImg)
    colorbar
    axis equal
    axis tight
    title('Clutter-Removed Image (CA-CFAR)')

    subplot(2, 3, 3)
    imagesc(20 * log10(resp_act / max(resp_act(:))))
    colorbar
    axis equal
    axis tight
    title('Predicted Response (CA-CFAR)')
 
    subplot(2, 3, 4)
    imagesc(clutterRemovedImgOSCFAR)
    colorbar
    axis equal
    axis tight
    title('Clutter-Removed Image (OS-CFAR)')

    subplot(2, 3, 5)
    imagesc(20 * log10(resp_actOSCFAR / max(resp_actOSCFAR(:))))
    colorbar
    axis equal
    axis tight
    title('Predicted Response (OS-CFAR)')
     

    fh.Position = fh.Position + [0 0 1000 400];
end

function plotDetectionVsSNR(detectionProbBefore, detectionProbAfter, detectionProbAfterOSCFAR, SNRdB)
    % Plot detection probability versus SNR

    figure;
    plot(SNRdB, detectionProbBefore, 'bo', 'LineWidth', 2)
    hold on
     plot(SNRdB, detectionProbAfter, 'ro', 'LineWidth', 2)
    plot(SNRdB, detectionProbAfterOSCFAR, 'ro', 'LineWidth', 2)
    hold off
    grid on
    legend('Before Clutter Removal', 'After Clutter Removal (OS-CFAR)')
    title('Detection Probability vs. SNR')
    xlabel('SNR (dB)')
    ylabel('Detection Probability')
end
