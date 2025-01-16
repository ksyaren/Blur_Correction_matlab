clc;
clear all;

%% Veri Yollarını Tanımlama
dataDir = 'test'; % Veri seti yolu
blurDir = fullfile(dataDir, 'blur');
sharpDir = fullfile(dataDir, 'sharp');

%% Veri Setlerini Yükleme
blurDS = imageDatastore(blurDir, 'IncludeSubfolders', true, 'FileExtensions', '.png');
sharpDS = imageDatastore(sharpDir, 'IncludeSubfolders', true, 'FileExtensions', '.png');

%% Veri Setini Eğitim ve Test Olarak Bölme
rng(42); % Rastgelelik için seed ayarla
splitRatio = 0.8; % %80 eğitim, %20 test
numImages = numel(blurDS.Files);
numTrain = round(splitRatio * numImages); % Eğitim için görüntü sayısı

% Rastgele indeksler oluştur
indices = randperm(numImages);
trainIndices = indices(1:numTrain);
testIndices = indices(numTrain+1:end);

% Eğitim ve test veri setlerini oluştur
blurTrainDS = subset(blurDS, trainIndices);
sharpTrainDS = subset(sharpDS, trainIndices);

blurTestDS = subset(blurDS, testIndices);
sharpTestDS = subset(sharpDS, testIndices);

%% Boyut Ayarı
inputSize = [224 224 3]; % Giriş boyutu: 224x224 piksel, 3 kanal (RGB)

%% Veri Hazırlama
% Eğitim Verileri
numTrainImages = numel(blurTrainDS.Files);
blurTrainImages = cell(numTrainImages, 1);
sharpTrainImages = cell(numTrainImages, 1);

for i = 1:numTrainImages
    blurImage = imresize(imread(blurTrainDS.Files{i}), inputSize(1:2));
    sharpImage = imresize(imread(sharpTrainDS.Files{i}), inputSize(1:2));
    blurTrainImages{i} = im2double(blurImage); % [0, 1] aralığına normalize etme
    sharpTrainImages{i} = im2double(sharpImage); % [0, 1] aralığına normalize etme
end

XTrain = cat(4, blurTrainImages{:});
YTrain = cat(4, sharpTrainImages{:});

% Test Verileri
numTestImages = numel(blurTestDS.Files);
blurTestImages = cell(numTestImages, 1);
sharpTestImages = cell(numTestImages, 1);

for i = 1:numTestImages
    blurImage = imresize(imread(blurTestDS.Files{i}), inputSize(1:2));
    sharpImage = imresize(imread(sharpTestDS.Files{i}), inputSize(1:2));
    blurTestImages{i} = im2double(blurImage); % [0, 1] aralığına normalize etme
    sharpTestImages{i} = im2double(sharpImage); % [0, 1] aralığına normalize etme
end

XTest = cat(4, blurTestImages{:});
YTest = cat(4, sharpTestImages{:});

%% Veri Artırma
function [augX, augY] = augmentData(X, Y)
    % Rastgele döndürme (-10 ila 10 derece arasında)
    rotationAngle = randi([-10, 10], 1);
    augX = imrotate(X, rotationAngle, 'crop');
    augY = imrotate(Y, rotationAngle, 'crop');
    
    % Rastgele yansıma
    if rand > 0.5
        augX = flip(augX, 2); % X ekseninde yansıma
        augY = flip(augY, 2);
    end
    
    % Rastgele ölçeklendirme (0.8 ile 1.2 arasında)
    scaleFactor = 0.8 + (1.2 - 0.8) * rand;
    augX = imresize(augX, scaleFactor);
    augY = imresize(augY, scaleFactor);
    augX = imresize(augX, size(X, 1:2)); % Orijinal boyuta döndür
    augY = imresize(augY, size(Y, 1:2));
end

% Eğitim Verisinde Veri Artırmayı Uygula
augXTrain = zeros(size(XTrain));
augYTrain = zeros(size(YTrain));

for i = 1:size(XTrain, 4)
    [augXTrain(:, :, :, i), augYTrain(:, :, :, i)] = augmentData(XTrain(:, ...
        :, :, i), YTrain(:, :, :, i));
end

%% U-Net Modelini Tanımlama
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')

    % Encoder
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1')
    batchNormalizationLayer('Name', 'batchnorm1')
    reluLayer('Name', 'relu1_1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_1')
    batchNormalizationLayer('Name', 'batchnorm2')
    reluLayer('Name', 'relu2_1')
    dropoutLayer(0.5, 'Name', 'dropout2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    % Bottleneck
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_1')
    batchNormalizationLayer('Name', 'batchnorm3')
    reluLayer('Name', 'relu3_1')
    dropoutLayer(0.5, 'Name', 'dropout3')

    % Decoder
    transposedConv2dLayer(2, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv2')
    reluLayer('Name', 'relu2_2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_2')
    batchNormalizationLayer('Name', 'batchnorm4')
    reluLayer('Name', 'relu2_3')

    transposedConv2dLayer(2, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv1')
    reluLayer('Name', 'relu1_2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_2')
    batchNormalizationLayer('Name', 'batchnorm5')
    reluLayer('Name', 'relu1_3')

    % Çıkış Katmanı
    convolution2dLayer(1, 3, 'Padding', 'same', 'Name', 'output_conv')
    sigmoidLayer('Name', 'sigmoid_output') % Sigmoid aktivasyonu
    regressionLayer('Name', 'output')
];
%% 

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% Modeli Eğitme
net = trainNetwork(augXTrain, augYTrain, layers, options);


%% Modeli Kaydetme
save('trainedUNetModel.mat', 'net');
disp('Model başarıyla kaydedildi.');