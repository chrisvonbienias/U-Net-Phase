clear
close all
clc

load('Dane_1.mat')


%% network architecture - CNN with one path

layers=[
    imageInputLayer([256 256 1],"Name","imageinput")
    convolution2dLayer([3 3],50,"Name","conv1_path1","Padding","same")
    reluLayer("Name","relu_1_path1")
    convolution2dLayer([3 3],50,"Name","conv2_path1","Padding","same")
    reluLayer("Name","relu_2_path1")
    convolution2dLayer([3 3],50,"Name","conv3_path1","Padding","same")
    reluLayer("Name","relu_3_path1")
    convolution2dLayer([3 3],50,"Name","conv4_path1","Padding","same")
    reluLayer("Name","relu_4_path1")
    convolution2dLayer([3 3],50,"Name","conv5_path1","Padding","same")
    reluLayer("Name","relu_5_path1")
    convolution2dLayer([3 3],50,"Name","conv6_path1","Padding","same")
    reluLayer("Name","relu_6_path1")
    %additionLayer(1,'Name','addition')
    
    convolution2dLayer([3 3],50,"Name","conv1_output1","Padding","same")
    reluLayer("Name","relu1_output1")
    convolution2dLayer([3 3],2,"Name","conv2_output1","Padding","same")
    reluLayer("Name","relu2_output1")
    regressionLayer("Name","regressionoutput1")];

lgraph=layerGraph(layers);

% layers=[
%     convolution2dLayer([3 3],50,"Name","conv1_output2","Padding","same")
%     reluLayer("Name","relu1_output2")
%     convolution2dLayer([3 3],1,"Name","conv2_output2","Padding","same")
%     reluLayer("Name","relu2_output2")
%     regressionLayer("Name","regressionoutput2")];
% 
% lgraph = addLayers(lgraph,layers);
% lgraph = connectLayers(lgraph,'relu_3_path1','conv1_output2');

%% alternative network architecture - CNN with more paths
imSize = [256,256,1];
paths = 10; %tu mo¿na zmieniaæ dopóki moce laptopa nie padn¹ i bêdzie siê coœ poprawia³o
layers=[
    imageInputLayer(imSize,"Name","imageinput","Normalization","none")];
lgraph=layerGraph(layers);
Fno = 16;
for ii=1:paths
    layers=[        
        
        convolution2dLayer([3 3],Fno,"Name","conv1_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_1_path"+num2str(ii))
        maxPooling2dLayer([ii ii],"Name","maxpoolForUnpool1_path"+num2str(ii),"HasUnpoolingOutputs",false,"Padding","same","Stride",[ii ii])
        convolution2dLayer([3 3],Fno,"Name","conv2_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_2_path"+num2str(ii))
        convolution2dLayer([3 3],Fno,"Name","conv3_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_3_path"+num2str(ii))
        convolution2dLayer([3 3],Fno,"Name","conv4_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_4_path"+num2str(ii))
        convolution2dLayer([3 3],Fno,"Name","conv5_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_5_path"+num2str(ii))
        resize2dLayer("Scale",256/ceil(256/ii),"Name","resize_path"+num2str(ii))
%         maxUnpooling2dLayer("Name","maxunpool1_path"+num2str(ii))
        convolution2dLayer([3 3],Fno,"Name","conv6_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_6_path"+num2str(ii));
        convolution2dLayer([3 3],Fno,"Name","conv7_path"+num2str(ii),"Padding","same")
        reluLayer("Name","relu_7_path"+num2str(ii))];
%         convolution2dLayer([3 3],Fno,"Name","conv8_path"+num2str(ii),"Padding","same")
%         reluLayer("Name","relu_8_path"+num2str(ii))];
        %additionLayer(1,'Name','addition')
        lgraph = addLayers(lgraph,layers);
end

layers=[
    additionLayer(paths,'Name','addition')
    convolution2dLayer([3 3],50,"Name","conv1_output1","Padding","same")
    reluLayer("Name","relu1_output1")
    convolution2dLayer([3 3],2,"Name","conv2_output1","Padding","same")
    reluLayer("Name","relu2_output1")
    segRegressionLayer("seg_regressionoutput1")];
%     regressionLayer("Name","regressionoutput1")];

lgraph = addLayers(lgraph,layers);
for ii=1:paths
    lgraph = connectLayers(lgraph,"relu_7_path"+num2str(ii),"addition/in"+num2str(ii));
    lgraph = connectLayers(lgraph,'imageinput',"conv1_path"+num2str(ii));
%     lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/indices","maxunpool1_path"+num2str(ii)+"/indices");
%     lgraph = connectLayers(lgraph,"maxpoolForUnpool1_path"+num2str(ii)+"/size","maxunpool1_path"+num2str(ii)+"/size");
end

%% another alternative - encoder-decoder architecture
num_encoder=200;
num_filters=4;
layers=[
    imageInputLayer([256 256 1],'Normalization','none',"Name","imageinput")
    %encoder
    convolution2dLayer([4 4],num_filters,"Padding","same","Name","conv1")%256x256 
    reluLayer("Name","conv_relu1")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool1","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],2*num_filters,"Padding","same","Name","conv2")%128x128
    reluLayer("Name","conv_relu2")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool2","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],4*num_filters,"Padding","same","Name","conv3")%64x64
    reluLayer("Name","conv_relu3")    
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool3","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],8*num_filters,"Padding","same","Name","conv4")%32x32
    reluLayer("Name","conv_relu4")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool4","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],16*num_filters,"Padding","same","Name","conv5")%16x16
    reluLayer("Name","conv_relu5")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool5","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],32*num_filters,"Padding","same","Name","conv6")%4x4
    reluLayer("Name","conv_relu6")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool6","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],64*num_filters,"Padding","same","Name","conv7")%2x2
    reluLayer("Name","conv_relu7")
    maxPooling2dLayer([2 2],"Name","maxpoolForUnpool7","HasUnpoolingOutputs",true,"Padding","same","Stride",[2 2])
    convolution2dLayer([4 4],num_encoder,"Padding","same","Name","conv8")%1x1
    reluLayer("Name","conv_relu8")
    %decoder
    convolution2dLayer([4 4],64*num_filters,"Padding","same",'Name','deconv1')%2x2
    reluLayer('Name','deconv_relu1')
    maxUnpooling2dLayer("Name","maxunpool7")
    convolution2dLayer([4 4],32*num_filters,"Padding","same",'Name','deconv2')%4x4
    reluLayer('Name','deconv_relu2')
    maxUnpooling2dLayer("Name","maxunpool6")
    convolution2dLayer([4 4],16*num_filters,"Padding","same",'Name','deconv3')%8x8
    reluLayer('Name','deconv_relu3')
    maxUnpooling2dLayer("Name","maxunpool5")
    convolution2dLayer([4 4],8*num_filters,"Padding","same",'Name','deconv4')%16x16
    reluLayer('Name','deconv_relu4')
    maxUnpooling2dLayer("Name","maxunpool4")
    convolution2dLayer([4 4],4*num_filters,"Padding","same",'Name','deconv5')%32x32
    reluLayer('Name','deconv_relu5')
    maxUnpooling2dLayer("Name","maxunpool3")
    convolution2dLayer([4 4],2*num_filters,"Padding","same",'Name','deconv6')%64x64
    reluLayer('Name','deconv_relu6')
    maxUnpooling2dLayer("Name","maxunpool2")
    convolution2dLayer([4 4],num_filters,"Padding","same",'Name','deconv7')%128x128
    reluLayer('Name','deconv_relu7')
    maxUnpooling2dLayer("Name","maxunpool1")
    convolution2dLayer([4 4],2,"Padding","same",'Name','deconv8')%256x256
    reluLayer('Name','deconv_relu8')
    regressionLayer("Name","regressionoutput1")%maeRegressionLayer('mae')%
    ];
    %additionLayer(1,'Name','addition')
    lgraph=layerGraph(layers);
    
for ii=1:7
    
    lgraph = connectLayers(lgraph,"maxpoolForUnpool"+num2str(ii)+"/indices","maxunpool"+num2str(ii)+"/indices");
    lgraph = connectLayers(lgraph,"maxpoolForUnpool"+num2str(ii)+"/size","maxunpool"+num2str(ii)+"/size");
end
%% training


options = trainingOptions('adam', ...
    'MiniBatchSize',1, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch',...
    'Plots','training-progress',...
    'ValidationData',{(inputValidationImages+3), (outputValidationImages)});
network = trainNetwork((inputTrainImages+3), (outputTrainImages),lgraph, options);

%% displaying results
close all
imi = 101;

Y = predict(network,inputTestImages(:,:,:,imi)+3);
figure; imagesc(inputTestImages(:,:,1,imi)+3); title('input'); colorbar
% figure; imagesc(inputTestImages(:,:,1,imi),[0 6]); title('input'); 
figure; imagesc(Y(:,:,1)+Y(:,:,2)); title('output 1 + output 2'); colorbar
figure; imagesc(Y(:,:,1)); title('output 1'); colorbar
figure; imagesc(outputTestImages(:,:,1,imi)); title('known output 1'); colorbar
figure; imagesc(Y(:,:,2)); title('output 2'); colorbar
figure; imagesc(outputTestImages(:,:,2,imi)); title('known output 2'); colorbar

%% 
close all
imi = 102;
nn = 128;
nn2 = 128;

figure;
tiledlayout(3,4,'TileSpacing','compact','Padding','compact')
Y = predict(network,inputTestImages(:,:,:,imi)+3);

nexttile(1)
t1 = inputTestImages(:,:,1,imi)-3;
imagesc(t1); title('input image'); colorbar; axis image
mi = min(t1(:)); ma = max(t1(:));
nexttile(5)
imagesc(Y(:,:,1)+Y(:,:,2)-6,[mi ma]); title('output 1 + output 2'); 
colorbar; axis image
nexttile(2)
t2 = outputTestImages(:,:,1,imi)-3;
imagesc(t2); title('ground truth 1'); colorbar; axis image; hold on
mi = min(t2(:)); ma = max(t2(:));
plot([1,256],[nn,nn],'--r','LineWidth',2); hold off
nexttile(6)
imagesc(Y(:,:,1)-3,[mi ma]); title('output 1'); colorbar; axis image
nexttile(3)
t3 = outputTestImages(:,:,2,imi)-3;
imagesc(t3); title('ground truth 2'); colorbar; axis image; hold on
plot([1,256],[nn2,nn2],'--r','LineWidth',2); hold off
mi = min(t3(:)); ma = max(t3(:));
nexttile(7)
imagesc(Y(:,:,2)-3,[mi ma]); title('output 2'); colorbar; axis image
nexttile(4)
plot(1:256,t1(nn,:),1:256,t2(nn,:),1:256,Y(nn,:,1)-3); grid on
legend('input','ground truth 1','output 1','Location','southwest')
title('Positive beads cross section')
nexttile(8)
plot(1:256,t1(nn2,:),1:256,t3(nn2,:),1:256,Y(nn2,:,2)-3); grid on
legend('input','ground truth 2','output 2','Location','southwest')
title('Negative beads cross section')
nexttile(9)
imagesc(t1-(Y(:,:,1)+Y(:,:,2)-6)); colorbar; axis image
tmp = t1-(Y(:,:,1)+Y(:,:,2)-6);
title(['input - (output1 + output2); RMSE = ',num2str(rms(tmp(:)))])
nexttile(10)
imagesc(t2-(Y(:,:,1)-3)); colorbar; axis image
tmp = t2-(Y(:,:,1)-3);
title(['ground truth 1 - output1; RMSE = ',num2str(rms(tmp(:)))])
nexttile(11)
imagesc(t3-(Y(:,:,2)-3)); colorbar; axis image
tmp = t3-(Y(:,:,2)-3);
title(['ground truth 2 - output2; RMSE = ',num2str(rms(tmp(:)))])
set(gcf,'Color','w')