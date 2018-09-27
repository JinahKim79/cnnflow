% Script to train CNN-Flow on the Middlebury dataset
% 3 versions are provided below, to train the entire network from scratch, or to train parts of it with other parts fixed to hard-coded suitable weights

gpu = 1 ; % ID of the GPU to use, set to 0 to run on CPU
stride = [1 1 4 4 4] ; % Absolute resolution of the feature maps at each layer, relatively to the original input (must be decreasing, except for the very last layer, where a bilinear upsampling is allowed)
leftRightSplit = true ; % Use the left/right halves of the images for training/validation

%--------------------------------------------------------------------------
% Train decoding only (motion filters + smoothing fixed)
%--------------------------------------------------------------------------
% Classification loss, no recurrent iteration
nRecurrentIterations = 1 ; % Only 1 (feedforward) iteration, no recurrent loop
flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',false,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 1 0],'lr',[0 0 .2 0],'lrBias',[0 0 .2 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',1,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[100 2 1]) ;

% Fine-tune decoding only, EPE loss
for nRecurrentIterations = 1:3 % Repeat for more and more recurrent iterations
  flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',true,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 0 0],'lr',[0 0 .1 0],'lrBias',[0 0 .1 0],'fixBordersSmoothing',false,'nEpochs',1000,'lossType',0,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[20 2 1]) ;
end

%--------------------------------------------------------------------------
% Same except train decoding + motion filters (smoothing fixed)
%--------------------------------------------------------------------------
% Classification loss, no recurrent iteration
nRecurrentIterations = 1 ; % Only 1 (feedforward) iteration, no recurrent loop
flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',false,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 1 0],'lr',[0 0 .2 0],'lrBias',[0 0 .2 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',1,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[100 2 1]) ;

% Fine-tune decoding only, EPE loss
for nRecurrentIterations = 1:3 % Repeat for more and more recurrent iterations
  flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',true,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 0 0],'lr',[.1 .1 0 0],'lrBias',[0 0 .1 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',0,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[20 2 1]) ;
end

%--------------------------------------------------------------------------
% Same except train everything: decoding + motion filters + smoothing
%--------------------------------------------------------------------------
% Classification loss, no recurrent iteration
nRecurrentIterations = 1 ; % Only 1 (feedforward) iteration, no recurrent loop
flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',false,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 1 0],'lr',[0 0 .2 0],'lrBias',[0 0 .2 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',1,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[100 2 1]) ;

% Fine-tune decoding only, EPE loss
  flow('modelId',1,'runTrVaTe',[1 20 0],'usegpu',gpu,'batchsize',15,'nFrames',3,'bidirectionalFlow',false,'maskBorders',true,'saveVisualizations',[0 0 0],'cropInput',[388 584],'leftRightSplit',leftRightSplit,'stride',stride,'continue',true,'nRecurrentIterations',nRecurrentIterations,'randomInit',[0 0 0 0],'lr',[1 .005 1 0],'lrBias',[0 0 1 0],'fixBordersSmoothing',false,'nEpochs',500,'lossType',0,'weightDecay',0,'reuseIntermediateResults',[1 1 0],'autoStop',[20 2 1]) ;
%end
