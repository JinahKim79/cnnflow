% Script to run a trained network on the test split of the Middlebury dataset
% This will load the provided pretrained network, or the result of demoCnnFlow_train if you ran it before
% The call to flow() with 'displayWeights'=true will also display the weights of the convlutional layers (both the "full" set of weigts and the "reduced" set from the enforced rotation invariance)

gpu = 1 ; % ID of the GPU to use, set to 0 to run on CPU
stride = [1 1 4 4 4] ; % Absolute resolution of the feature maps at each layer, relatively to the original input (must be decreasing, except for the very last layer, where a bilinear upsampling is allowed)
nRecurrentIterations = 3 ; % Usually 3 is best (more makes noisy results, fewer is less accurate; ideally you should use the same number for testing as used for training)

% Evaluate the network on the validation (=training in our case) and test splits
flow('modelId',1,'runTrVaTe',[0 1 1],'usegpu',gpu,'batchsize',1,'persistentBatchLoader',false,'nFrames',3,'bidirectionalFlow',false,'maskBorders',false,'saveVisualizations',[1 1 1],'cropInput',false,'leftRightSplit',false,'stride',stride,'continue',true,'nRecurrentIterations',nRecurrentIterations,'fixBordersSmoothing',true,'displayWeights',true) ;
