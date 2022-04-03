clear
close all
clc

disp('Loading dataset...')
load('Dane_1.mat')

%% Network architecture

disp('Creating architecture...')
u_net_arch

%% Training

disp('Training...')
options = trainingOptions('adam', ...
    'MiniBatchSize',1, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'ValidationData',{(inputValidationImages+3), (outputValidationImages)});

network = trainNetwork((inputTrainImages+3), (outputTrainImages),unet_graph, options);

%% Results

disp('Displaying results...')
close all
imi = 101;

Y = predict(network,inputTestImages(:,:,:,imi)+3);
figure; imagesc(inputTestImages(:,:,1,imi)+3); title('input'); colorbar
figure; imagesc(Y(:,:,1)+Y(:,:,2)); title('output 1 + output 2'); colorbar
figure; imagesc(Y(:,:,1)); title('output 1'); colorbar
figure; imagesc(outputTestImages(:,:,1,imi)); title('known output 1'); colorbar
figure; imagesc(Y(:,:,2)); title('output 2'); colorbar
figure; imagesc(outputTestImages(:,:,2,imi)); title('known output 2'); colorbar