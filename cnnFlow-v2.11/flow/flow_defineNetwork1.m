function net = flow_defineNetwork1(p)
%FLOW_DEFINENETWORK Define CNN for optical flow estimation; multiscale, with recurrent (warping) connection.

net = {} ;

% Define separate subnets for each scale
p.downscalings = sort(p.downscalings, 'descend') ; % Sort scales from coarse to fine (arbitrary choice)
for scaleId = 1:p.nScales
  n = length(net) + 1 ; % ID of the current subnet
  currentDownscaling = p.downscalings(scaleId) ; % Downscaling factor (>= 1)

  %------------------------------------------------------------------------
  % Subnet input and preprocessing
  %------------------------------------------------------------------------
  if n == 1 % First subnet
    net{n}{1} = struct('type', 'getBatch', 'getBatchFcn', @flow_getBatch) ;
    net{n}{end+1} = struct('type', 'warp', 'uvMapSrc', [p.nScales, -1], 'uvMapUpscalingFactor', p.stride(4)/p.stride(1)) ; % Warp for doing multiple rounds
  else
    net{n}{1} = struct('type', 'copy', 'src', [1 2]) ;
    net{n}{end+1} = struct('type', 'identity') ;
  end

  net{n}{end+1} = struct('type', 'downsize', 'scaleFactor', 1/currentDownscaling) ;

  % Center-surround filter (substract local DC component)
  w = 7 ; % Filter width
  f = getGaussian2D(w, 2.27) ;
  filters = zeros(w, w, p.nFrames, p.nFrames, 'single') ;
  for i = 1:p.nFrames, filters(:, :, i, i) = single(f) ; end
  net{n}{end+1} = struct('type', 'substractconv', 'filters', single(filters), 'biases', zeros(1, size(filters, 4), 'single'), 'stride', 1, 'pad', floor(size(f, 1) / 2), 'filtersLearningRate', 0, 'biasesLearningRate', 0, 'filtersWeightDecay', 0, 'biasesWeightDecay', 0) ;

  % Std-normalize
  if p.enableStdNormalization
    net{n}{end+1} = struct('type', 'stdnormalization', 'windowSize', ceil(w*2/3)) ;
  else
    net{n}{end+1} = struct('type', 'identity') ; % Fill the spot with a dummy layer to keep the number of layers the same
  end

  %------------------------------------------------------------------------
  % Motion filters
  %------------------------------------------------------------------------
  if p.randomInit(1) % Random initialization
    filters = p.randomInit(1) * initFilters(9, 9, p.nFrames, p.nSpeeds*p.nOris) ;
    filters = filters(:, :, 1:p.nFrames, 1:p.nSpeeds*1) ; % Keep 1 orientation
  else % Hard-coded filters
    filters = zeros(9, 9, p.nFrames, p.nSpeeds, 1) ;
    for s = 1:p.nSpeeds
      [h1, h2] = makeMotionFilter(p.nFrames, 0, -p.speeds(s), 1, 1, p.bidirectionalFlow, false, false) ;
      filters(:, :, :, s, 1) = h1 ;
    end
    filters = reshape(filters, size(filters, 1), size(filters, 2), p.nFrames, p.nSpeeds*1) ;
  end
  filters = crop(filters, p.motionFilterWidth, p.motionFilterWidth, 'center') ; % Crop filters spatially

  P = sparse(getRotatorMatrix(size(filters, 1), p.nFrames, p.nSpeeds, p.nOris, false, false)) ;
  net{n}{end+1} = struct('type', 'ori-conv', 'filters', single(filters), 'biases', zeros(1, size(filters, 4), 'single'), 'stride', p.stride(2)/p.stride(1), 'pad', (size(filters, 1)-1)/2, 'filtersLearningRate', p.lr(1), 'biasesLearningRate', p.lrBias(1), 'filtersWeightDecay', p.wd(1), 'biasesWeightDecay', 0, 'doNotCopy', p.randomInit(1), 'nOrientations', p.nOris, 'weightProjection', P, 'filterSize', size(filters, 1), 'nChannelsIn', p.nFrames, 'nChannelsOut', p.nSpeeds) ;

  net{n}{end+1} = struct('type', 'square') ; % Rectification

  % Pool for phase invariance
  poolingSize = p.stride(3)/p.stride(2) ;
  %net{n}{end+1} = struct('type', 'pool', 'pool', poolingSize, 'method', 'max', 'pad', floor(poolingSize-1)/2), 'stride', poolingSize) ; % Symmetric (on all sides) padding
  net{n}{end+1} = struct('type', 'pool', 'pool', poolingSize, 'method', 'max', 'pad', [0 (poolingSize-1) 0 (poolingSize-1)], 'stride', poolingSize) ; % Non-symmetric padding (right/bottom only) to emulate Caffe's pooling behaviour

  % Normalization
  switch p.channelNormalization
    case 'L1', net{n}{end+1} = struct('type', 'normalizel1', 'nGroups', p.nOris, 'biases', 1e-8, 's', 1, 'groupDim', 3, 'biasesLearningRate', 0, 'biasesWeightDecay', 0) ;
    case 'L2', net{n}{end+1} = struct('type', 'normalizel2', 'nGroups', p.nOris, 'e', 1e-8, 'groupDim', 3) ;
    case 'none', net{n}{end+1} = struct('type', 'identity') ;
    otherwise, error('Unknown p.channelNormalization') ;
  end

  %------------------------------------------------------------------------
  % Gaussian smoothing
  %------------------------------------------------------------------------
  w = ceil(16 / p.stride(3)) ; % Kernel radius
  h = getGaussian2D(w*2+1, 0.6, true) ; % Create Gaussian kernel

  if p.fixBordersSmoothing
    net{n}{end+1} = struct('type', 'padborders', 'w', w) ; % Add mirror padding before applying smoothing filters
  else
    net{n}{end+1} = struct('type', 'identity') ;
  end

  % Arbitrary kernels
  %%{
  if p.randomInit(2) % Random initialization
    filters = p.randomInit(2) * initFilters(w*2+1, w*2+1, p.nSpeeds*p.nOris, p.nSpeeds*p.nOris) ;
    filters = filters(:, :, 1:p.nSpeeds, 1:p.nSpeeds) ; % Keep 1 orientation
  else % Hardcoded Gaussians
    filters = zeros(2*w+1, 2*w+1, p.nSpeeds, p.nSpeeds) ;
    for s = 1:p.nSpeeds
      filters(:, :, s, s) = single(h) ;
    end
    filters = reshape(filters, 2*w+1, 2*w+1, p.nSpeeds, p.nSpeeds) ;
  end
  P = sparse(getRotatorMatrix(size(filters, 1), p.nSpeeds, p.nSpeeds, p.nOris, false, true)) ;
  net{n}{end+1} = struct('type', 'ori-conv', 'filters', single(filters), 'biases', zeros(1, size(filters, 4), 'single'), 'stride', p.stride(4)/p.stride(3), 'pad', (~p.fixBordersSmoothing * w), 'filtersLearningRate', p.lr(2), 'biasesLearningRate', p.lrBias(2), 'filtersWeightDecay', p.wd(2), 'biasesWeightDecay', 0, 'doNotCopy', p.randomInit(2), 'nOrientations', p.nOris, 'weightProjection', P, 'filterSize', w*2+1, 'nChannelsIn', p.nSpeeds, 'nChannelsOut', p.nSpeeds) ;
  %}

  % Kernels constrained to be isotropic
  %{
  if p.randomInit(2) % Random initialization
    filters = p.randomInit(2) * initFilters(w*2+1, w*2+1, p.nSpeeds*p.nOris, p.nSpeeds*p.nOris) ;
    filters = filters(1, 1:w+1, 1:p.nSpeeds, 1:p.nSpeeds) ; % Keep 1 cross-section and 1 orientation
  else % Hardcoded Gaussians
    filters = zeros(1, w+1, p.nSpeeds, p.nSpeeds) ;
    for s = 1:p.nSpeeds
      filters(1, :, s, s) = single(h(w+1, w+1:end)) ;
    end
    filters = reshape(filters, 1, w+1, p.nSpeeds, p.nSpeeds) ;
  end
  P = sparse(getIsotropizingMatrix(w*2+1, p.nSpeeds, p.nSpeeds, p.nOris)) ;
  net{n}{end+1} = struct('type', 'ori-conv', 'filters', single(filters), 'biases', zeros(1, size(filters, 4), 'single'), 'stride', p.stride(4)/p.stride(3), 'pad', (~p.fixBordersSmoothing * w), 'filtersLearningRate', p.lr(2), 'biasesLearningRate', p.lrBias(2), 'filtersWeightDecay', p.wd(2), 'biasesWeightDecay', 0, 'doNotCopy', p.randomInit(2), 'nOrientations', p.nOris, 'weightProjection', P, 'filterSize', w*2+1, 'nChannelsIn', p.nSpeeds, 'nChannelsOut', p.nSpeeds) ;
  %}

  %------------------------------------------------------------------------
  % Concatenate with the feature maps from previous scale
  %------------------------------------------------------------------------
  net{n}{end+1} = struct('type', 'upsize', 'newFactorOfOriginalInputDimensions', 1/p.stride(4), 'scaleFactor', currentDownscaling, 'scaleGradient', 2) ; % Upsample to a common resolution
  if n > 1 % Except first subnet
    net{n}{end+1} = struct('type', 'ori-concat', 'src', [n-1, length(net{n-1})], 'nOris', p.nOris) ; % Concatenate
  end
end % For each scale-specific subnet

%--------------------------------------------------------------------------
% Decoding (pixelwise weights)
%--------------------------------------------------------------------------
filters = p.randomInit(3) * initFilters(1, 1, p.nSpeeds*p.nScales*(p.nOris/2+1), p.nSpeedsDecoding*1) ;
P = sparse(getCirculanzingMatrix(p.nSpeeds*p.nScales, p.nSpeedsDecoding, p.nOris, true)) ;
net{n}{end+1} = struct('type', 'ori-conv', 'filters', single(filters), 'biases', zeros(1, size(filters, 4), 'single'), 'stride', 1, 'pad', 0, 'filtersLearningRate', p.lr(3), 'biasesLearningRate', p.lrBias(3), 'filtersWeightDecay', p.wd(3), 'biasesWeightDecay', 0, 'doNotCopy', p.randomInit(3), 'nOrientations', p.nOris, 'weightProjection', P, 'filterSize', 1, 'nChannelsIn', p.nSpeeds*p.nScales, 'nChannelsOut', p.nSpeedsDecoding) ;

%--------------------------------------------------------------------------
% Final part (result and loss)
%--------------------------------------------------------------------------
% Mask the values near the borders (less accurte because of padding/border effects) so that they are not used for backpropagation
if p.maskBorders
  net{n}{end+1} = struct('type', 'clearborder', 'width', ceil(16 / p.stride(4)), 'newValue', NaN) ;
else
  net{n}{end+1} = struct('type', 'identity') ;
end

weightsLayer = length(net{n}) ; % Keep the index of the current layer to be linked to the classification loss later

net{n}{end+1} = struct('type', 'softmax') ;

% Generate the (hardocded) weights that project the softmax values (classification) onto U/V components (regression)
filters = zeros(1, 1, p.nSpeedsDecoding, p.nOris, 2) ;
for o2 = 1:p.nOris, for s = 1:p.nSpeedsDecoding
  filters(1, 1, s, o2, 1:2) = [cos(p.oris(o2)), sin(p.oris(o2)) ] .* p.speedsDecoding(s) ;
end, end
filters = reshape(filters, 1, 1, p.nSpeedsDecoding*p.nOris, 2) ;
net{n}{end+1} = struct('type', 'conv', 'filters', single(filters), 'biases', zeros(1, 2, 'single'), 'stride', 1, 'pad', 0, 'filtersLearningRate', p.lr(4), 'biasesLearningRate', p.lrBias(4), 'filtersWeightDecay', p.wd(4), 'biasesWeightDecay', 0, 'doNotCopy', p.randomInit(4)) ;

% Recurrent connection
net{n}{end+1} = struct('type', 'flowRecurrent', 'backTo', [1 2], 'nRecurrentIterations', p.nRecurrentIterations, 'sumResults', true) ;
net{1}{2}.uvMapSrc(2) = length(net{n}) ; % Link up the warp layer to the result (the UV map used to warp)

if p.stride(5) < p.stride(4) % Upsize if output is required at a finer resolution
  net{n}{end+1} = struct('type', 'upsize', 'newFactorOfOriginalInputDimensions', 1/p.stride(5), 'scaleFactor', p.stride(4) / p.stride(5), 'scaleGradient', 0) ;
end

if p.stride(1) > 1 % If we worked on a downsampled input, we detected smaller motions than what was in the original frames, so we scale up the UV map values
  net{n}{end+1} = struct('type', 'scale', 's', p.stride(1)) ;
end

net{end}{1}.outputLayer = numel(net{end}) ; % Mark this layer as the one containing the values of interest (before the loss)

switch p.lossType
  case 0 % Regression (EPE) loss
    net{n}{end+1} = struct('type', 'epeloss') ;
  case 1 % Classification (log) loss
    net{n}{end+1} = struct('type', 'flowloss', 'outputValuesSrc', [length(net), weightsLayer+2], 'src', [length(net), weightsLayer]) ;
  otherwise
    error('Unknown p.lossType') ;
end
