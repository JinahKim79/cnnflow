% Train a simple model with Matlab that can be saved as a Caffe model

% Simplifications versus our complete Matlab model: (because of limitations of the Caffe layers)
% - no recurrent iterations (nRecurrentIterations=1)
% - reduced set of scales (see downscalings below)
% - no initial normalization of images with local standard deviation (enableStdNormalization=0)
% - sped up training with fixed motion filters (3D Gaussian derivatives, randomInit(1)=1) and smoothing filters (isotropic Gaussians, randomInit(2)=0)
% - no fixing of the borders when downsampling/upsampling/smoothing the feature maps (the Matlab version normalizes the values at the borders and/or adds mirror padding)
% - normalization across channels of a same orientation: L2 instead of L1

gpu = 1 ; % ID of the GPU to use, set to 0 to run on CPU
leftRightSplit = true ; % Use the left/right halves of the images for training/validation
stride = [1 1 4 4 4] ; % Absolute resolution of the feature maps at each layer, relatively to the original input (must be decreasing, except for the very last layer, where a bilinear upsampling is allowed)

% Define the dimensions of input images to be used with the Caffe model
inputSz = [256, 256] ;

% Automatically determine a set of scales that work with the chosen input dimensions (so that downsampling then upsampling by these integer factors keep the original size)
downscalings = 1:48 ; % Candidate scaling factors
validDownscalings = (  downscalings .* ceil( ceil(inputSz(1) / stride(4)) ./ downscalings )  ==  ceil(inputSz(1)/stride(4))  ) ...
                  & (  downscalings .* ceil( ceil(inputSz(2) / stride(4)) ./ downscalings )  ==  ceil(inputSz(2)/stride(4))  ) ;
downscalings = downscalings(validDownscalings)

% Or, alternatively, manually pick a set of scales
%downscalings = [1 2 4 8 16] ;
downscalings = [1 2 4 8] ;

%--------------------------------------------------------------------------

% Train with classification loss
flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',false,'nRecurrentIterations',1,'randomInit',[0 0 1 0],'lr',[0 0 .2 0],'lrBias',[0 0 .2 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',1,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[100 2 1],'downscalings',downscalings,'enableStdNormalization',false,'channelNormalization','L2','dataAugmentation',0) ;

% Fine-tune with EPE loss
flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',true,'nRecurrentIterations',1,'randomInit',[0 0 0 0],'lr',[0 0 .1 0],'lrBias',[0 0 .1 0],'fixBordersSmoothing',false,'nEpochs',1000,'lossType',0,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[100 2 1],'downscalings',downscalings,'enableStdNormalization',false,'channelNormalization','L2','dataAugmentation',0) ;

% Test (can skip)
%flow('modelId',1,'runTrVaTe',[0 1 1],'usegpu',gpu,'batchsize',1,'persistentBatchLoader',false,'nFrames',3,'bidirectionalFlow',false,'maskBorders',false,'saveVisualizations',[1 1 1],'cropInput',false,'leftRightSplit',false,'stride',stride,'continue',true,'nRecurrentIterations',1,'fixBordersSmoothing',true,'displayWeights',true,'downscalings',downscalings,'enableStdNormalization',false,'channelNormalization','L2') ;

% Convert to a Caffe model
inputSz = [256, 256] ; % Define the dimensions of input images (used as input by the caffe model)
load(fullfile('results', 'model1', 'net.mat'))
flow_saveAsCaffeNetwork(p, net, inputSz, 'cnnFlow', true)
