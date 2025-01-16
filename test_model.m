% Modeli yükle
load('trainedUNetModel.mat');
disp('Model başarıyla yüklendi.');


% Modeli test verisi üzerinde değerlendirme
YPred = predict(net, XTest);
%% 
% Rastgele bir test görüntüsü seçme
numTestImages = size(XTest, 4); % Test setindeki toplam görüntü sayısını al
testIdx = randi(numTestImages); % Rastgele bir test görüntüsü seçin
blurTest = XTest(:, :, :, testIdx); % Bulanık test görüntüsü
sharpTest = YTest(:, :, :, testIdx); % Gerçek keskin görüntü

% Eğer bulanık test görüntüsü grayscale ise RGB'ye dönüştür
if size(blurTest, 3) == 1
    blurTest = cat(3, blurTest, blurTest, blurTest); % Grayscale'i RGB'ye çevir
end

% Model ile tahmin yapma
predictedImage = predict(net, blurTest); % Model bulanık görüntüyü netleştirir

% Tahmini normalize edip RGB'ye dönüştür
predictedImage = mat2gray(predictedImage); % Normalize (0-1 aralığına getir)
if size(predictedImage, 3) == 1
    predictedImage = cat(3, predictedImage, predictedImage, predictedImage); % Grayscale'i RGB'ye çevir
end
predictedImage = uint8(255 * predictedImage); % [0, 255] aralığına çevir

% Gradyan tabanlı keskinleştirme işlemi
dimage = im2double(sharpTest); % Keskin görüntüyü double formatına çevir
h1 = fspecial('gaussian', [3 3], 1); % 3x3 Gaussian filter
h2 = fspecial('gaussian', [5 5], 2); % 5x5 Gaussian filter
gradient = convn(dimage, h1, "same") - convn(dimage, h2, "same");
amount = 2 * std2(gradient); % Gradyanın standart sapmasına göre ayarlanır.
sharpened = dimage + amount .* gradient;
sharpened = uint8(255 * mat2gray(sharpened)); % [0, 255] aralığına çevir

% Görüntüleri yan yana gösterme
figure;
subplot(1, 3, 1);
imshow(blurTest);
title('Bulanık Görüntü');
subplot(1, 3, 2);
imshow(sharpTest);
title('Gerçek Keskin Görüntü');
subplot(1, 3, 3);
imshow(sharpened);
title('Modelin Tahmini');

%% 

% Performans metrikleri (PSNR ve SSIM)
psnrValues = zeros(size(XTest, 4), 1);
ssimValues = zeros(size(XTest, 4), 1);

for i = 1:size(XTest, 4)
    % Görüntüleri uint8 tipine dönüştür
    YPred_uint8 = im2uint8(YPred(:, :, :, i));
    YTest_uint8 = im2uint8(YTest(:, :, :, i));
    
    % PSNR ve SSIM hesapla
    psnrValues(i) = psnr(YPred_uint8, YTest_uint8);
    ssimValues(i) = ssim(YPred_uint8, YTest_uint8);
end

fprintf('Ortalama PSNR: %.2f dB\n', mean(psnrValues));
fprintf('Ortalama SSIM: %.4f\n', mean(ssimValues));