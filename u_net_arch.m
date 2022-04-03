%% Architecture - U-Net
init_conv = 32;
unet_graph = layerGraph();

% First stage encoder
tempLayers = [
    imageInputLayer([256 256 1],"Name","ImageInputLayer")
    convolution2dLayer([3 3],init_conv, ...
    "Name","Encoder-Stage-1-Conv-1","Padding", ...
    "same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],init_conv, ...
    "Name","Encoder-Stage-1-Conv-2","Padding", ...
    "same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
unet_graph = addLayers(unet_graph,tempLayers);

% Second stage encoder
tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-1-MaxPool")
    convolution2dLayer([3 3],init_conv*2, ...
    "Name","Encoder-Stage-2-Conv-1","Padding", ...
    "same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],init_conv*2, ...
    "Name","Encoder-Stage-2-Conv-2","Padding", ...
    "same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
unet_graph = addLayers(unet_graph,tempLayers);

% Third stage encoder
tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-2-MaxPool")
    convolution2dLayer([3 3],init_conv*4,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],init_conv*4,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
unet_graph = addLayers(unet_graph,tempLayers);

% Fourth stage encoder
tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-3-MaxPool")
    convolution2dLayer([3 3],init_conv*8,"Name","Encoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],init_conv*8,"Name","Encoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-4-ReLU-2")];
unet_graph = addLayers(unet_graph,tempLayers);

% Fifth stage encoder
tempLayers = [
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-4-MaxPool")
    convolution2dLayer([3 3],init_conv*16,"Name","Encoder-Stage-5-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-5-ReLU-1")
    convolution2dLayer([3 3],init_conv*16,"Name","Encoder-Stage-5-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Encoder-Stage-5-ReLU-2")];
unet_graph = addLayers(unet_graph,tempLayers);

% Bridge
tempLayers = [
    dropoutLayer(0.5,"Name","Encoder-Stage-5-DropOut")
    maxPooling2dLayer([2 2],"Name","Encoder-Stage-5-MaxPool")
    convolution2dLayer([3 3],init_conv*32,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-1")
    convolution2dLayer([3 3],init_conv*32,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Bridge-ReLU-2")
    dropoutLayer(0.5,"Name","Bridge-DropOut")
    transposedConv2dLayer([2 2],init_conv*16,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
unet_graph = addLayers(unet_graph,tempLayers);

% First stage decoder
tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-1-DepthConcatenation")
    convolution2dLayer([3 3],init_conv*16,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution2dLayer([3 3],init_conv*16,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv2dLayer([2 2],init_conv*8,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU")];
unet_graph = addLayers(unet_graph,tempLayers);

% Second stage decoder
tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-2-DepthConcatenation")
    convolution2dLayer([3 3],init_conv*8,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution2dLayer([3 3],init_conv*8,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv2dLayer([2 2],init_conv*4,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
unet_graph = addLayers(unet_graph,tempLayers);

% Third stage decoder
tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-3-DepthConcatenation")
    convolution2dLayer([3 3],init_conv*4,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution2dLayer([3 3],init_conv*4,"Name","Decoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    transposedConv2dLayer([2 2],init_conv*2,"Name","Decoder-Stage-4-UpConv","BiasLearnRateFactor",2,"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-UpReLU")];
unet_graph = addLayers(unet_graph,tempLayers);

% Fourth stage decoder
tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-4-DepthConcatenation")
    convolution2dLayer([3 3],init_conv*2,"Name","Decoder-Stage-4-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-1")
    convolution2dLayer([3 3],init_conv*2,"Name","Decoder-Stage-4-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-4-ReLU-2")
    transposedConv2dLayer([2 2],init_conv,"Name","Decoder-Stage-5-UpConv","BiasLearnRateFactor",2,"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-5-UpReLU")];
unet_graph = addLayers(unet_graph,tempLayers);

% Fifth stage decoder
tempLayers = [
    depthConcatenationLayer(2,"Name","Decoder-Stage-5-DepthConcatenation")
    convolution2dLayer([3 3],init_conv,"Name","Decoder-Stage-5-Conv-1","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-5-ReLU-1")
    convolution2dLayer([3 3],init_conv,"Name","Decoder-Stage-5-Conv-2","Padding","same","WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-5-ReLU-2")
    convolution2dLayer([1 1],2,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    regressionLayer("Name","regressionoutput")];
unet_graph = addLayers(unet_graph,tempLayers);

% clean up helper variable
clear tempLayers;

% connect layers
unet_graph = connectLayers(unet_graph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-5-DepthConcatenation/in2");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-4-DepthConcatenation/in2");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-MaxPool");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-3-ReLU-2","Decoder-Stage-3-DepthConcatenation/in2");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-4-ReLU-2","Encoder-Stage-4-MaxPool");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-4-ReLU-2","Decoder-Stage-2-DepthConcatenation/in2");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-5-ReLU-2","Encoder-Stage-5-DropOut");
unet_graph = connectLayers(unet_graph,"Encoder-Stage-5-ReLU-2","Decoder-Stage-1-DepthConcatenation/in2");
unet_graph = connectLayers(unet_graph,"Decoder-Stage-1-UpReLU","Decoder-Stage-1-DepthConcatenation/in1");
unet_graph = connectLayers(unet_graph,"Decoder-Stage-2-UpReLU","Decoder-Stage-2-DepthConcatenation/in1");
unet_graph = connectLayers(unet_graph,"Decoder-Stage-3-UpReLU","Decoder-Stage-3-DepthConcatenation/in1");
unet_graph = connectLayers(unet_graph,"Decoder-Stage-4-UpReLU","Decoder-Stage-4-DepthConcatenation/in1");
unet_graph = connectLayers(unet_graph,"Decoder-Stage-5-UpReLU","Decoder-Stage-5-DepthConcatenation/in1");