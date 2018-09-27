function [res, net] = evalCnn(p, net, batch, res, isTraining, reuseIntermediateResults)
%EVALCNN Evaluate CNN forward and backward.

% Authors: Andrea Vedaldi, Damien Teney

% Check input
assert(iscell(net)) ;
if isempty(res)
  res = cell(1, numel(net)) ;
  actuallyReuseIntermediateResults = false ; % Can't reuse intermediate results: they're not provided (e.g. 1st SGD iteration)
else
  actuallyReuseIntermediateResults = reuseIntermediateResults ;
end

if p.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

% Reinitialize the field 'nShared' (used to count accumulated derivatives with iterations/recurrent connections and shared weights)
for n = 1:length(net)
  if ~isempty(res{n})
    for l = 1:length(res{n})
      if isfield(res{n}(l), 'nShared')
        res{n}(l).nShared = [] ;
      end
    end
  end
end

n = 1 ; l = 1 ; % Start at the first subnet, first layer
backTo = [1 1] ; % Indices of subnet/layer to go back to during the backward pass; if different than [1 1], means we have a recurrent connection, and another forward+backward evaluation is performed when reaching it
recurrentIteration = 0 ; % Initialize counter of recurrent iterations (1-based: will be incremented at the beginning of the loop)

while true % For each iteration of a recurrent connection (performs multiple sets of forward+backward evaluations)
  recurrentIteration = recurrentIteration + 1 ;

  %------------------------------------------------------------------------
  % Forward
  %------------------------------------------------------------------------
  while true % For each subnetwork/layer ('while' instead of a 'for' so that the indices can be changed by code inside the loop to skip or go back iterations)
    %disp([n, l]) ; % Debug display

    % Initialize data structure to store the results of the evaluation of each layer
    if isempty(res{n})
      % Empty initialization
      nLayers = numel(net{n}) ;
      res{n} = struct(...
        'x', cell(1, nLayers + 1), ...
        'dzdx', cell(1, nLayers + 1), ...
        'dzdw', cell(1, nLayers + 1), ...
        'nShared', cell(1, nLayers + 1), ...
        'aux', cell(1, nLayers + 1), ...
        'time', num2cell(zeros(1, nLayers + 1)), ...
        'backwardTime', num2cell(zeros(1, nLayers + 1))) ;
    end

    ly = net{n}{l} ;

    % Check wether we can skip the forward evaluation of the current layer
    if actuallyReuseIntermediateResults && (l <= net{n}{1}.earlyStopMaxLayer)
      %fprintf('Skipping evaluation #%d.%d\n', n, l) ; % Debug display
      % Goes directly to the next layer
      if (l == numel(net{n})) && (n == numel(net)) % Reached the last layer of the last subnet
        break ;
      elseif l == numel(net{n}) % Reached the last layer of a subnet (not the last one)
        n = n + 1 ; l = 1 ; % Move to the first layer of the next subnet
      else
        l = l + 1 ; % Next layer, same subnet
      end
      continue ;
    end

    timer = tic() ;
    switch ly.type
      case 'getBatch'
        enableAugmentation = (p.dataAugmentation == 2) || ((p.dataAugmentation == 1) && isTraining) ; % If == 1, enable augmentation at training time, if == 2, enable augmentation at training AND validation/test time
        [res{n}(l+1).x, res{1}(1).labels, originalInputDimensions] = ly.getBatchFcn(p, batch, enableAugmentation) ;
        res{n}(l+1).originalInputDimensions = originalInputDimensions ;
      case 'identity'
        res{n}(l+1).x = res{n}(l).x ;
      case 'copy'
        res{n}(l+1).x = res{ly.src(1)}(ly.src(2)+1).x ;
        %if p.conserveMemory, res{ly.src(1)}(ly.src(2)+1).x = [] ; end % Clear the source: OK only if not used several times !
      case 'substractconv'
        res{n}(l+1).x = vl_nnconv(res{n}(l).x, ly.filters, ly.biases, 'pad', ly.pad, 'stride', ly.stride) ;
        % Creat a normalization map to account for border effects
        if ~isfield(net{n}{l}, 'normalizationMap') || ~isequal(size(res{n}(l+1).x), size(net{n}{l}.normalizationMap)) % Need to be computed once only, so save it in as a field of the layer
          net{n}{l}.normalizationMap = ones(size(res{n}(l).x, 1), size(res{n}(l).x, 2), size(res{n}(l).x, 3), 1, 'like', res{n}(l).x) ; % Create a matrix of ones of the same type as X
          net{n}{l}.normalizationMap = vl_nnconv(net{n}{l}.normalizationMap, ly.filters, [], 'pad', ly.pad, 'stride', ly.stride) ; % Get the 'contribution' of the filter to each pixel (smaller near the borders)
        end
        res{n}(l+1).x = bsxfun(@rdivide, res{n}(l+1).x, net{n}{l}.normalizationMap) ; % Fix the borders
        res{n}(l+1).x = res{n}(l).x - res{n}(l+1).x ; % Return the substraction of the convolution result from the original signal
        clear empty ;
      case 'ori-conv'
        if ~isfield(ly, 'allFilters') || (isfield(ly, 'allFiltersValid') && ~ly.allFiltersValid)
          % Generate all filters
          if p.useGpu && issparse(ly.weightProjection) && verLessThan('matlab', '8.5') % Sparse arrays on GPU not supported in Matlab pre-R2015a
            ly.allFilters = ly.weightProjection * double(gather(ly.filters(:))) ; % Transfer to CPU then run sparse multiplication; ly.weightProjection is sparse and therefore double (Matlab does not support sparse singles)
            ly.allFilters = gpuArray(single(ly.allFilters)) ; % Transfer result back to GPU
          else
            ly.allFilters = ly.weightProjection * double(ly.filters(:)) ;
            ly.allFilters = single(ly.allFilters) ;
          end
          ly.allFilters = reshape(ly.allFilters, ly.filterSize, ly.filterSize, size(res{n}(l).x, 3), ly.nChannelsOut * ly.nOrientations) ;
          % Generate all biases
          ly.allBiases = reshape(repmat(ly.biases, 1, 1, ly.nOrientations), 1, []) ; % Replicte the same biases for all orientations
          % Save (cache) the generated filters/biases
          net{n}{l}.allFilters = ly.allFilters ;
          net{n}{l}.allBiases = ly.allBiases ;
          net{n}{l}.allFiltersValid = true ;
        end % Otherwise rotated versions were already generated/available

        assert(size(res{n}(l).x, 3) == size(ly.allFilters, 3)) ; % Check there are no filter groups (not often used, usually a mistake !)
        res{n}(l+1).x = vl_nnconv(res{n}(l).x, ly.allFilters, ly.allBiases, 'pad', ly.pad, 'stride', ly.stride) ; % Perform the convolution
      case 'ori-maxpool'
        keyboard % TODO CHECK
        sz = [ size(res{n}(l).x, 1), size(res{n}(l).x, 2), size(res{n}(l).x, 3), size(res{n}(l).x, 4) ] ;
        res{n}(l).x = reshape(res{n}(l).x, sz(1), sz(2), sz(3) / ly.nOrientations, ly.nOrientations, sz(4)) ; % Expand the dimension of the orientation groups (modify res{n}(l).x in-place !)
        res{n}(l+1).x = max(res{n}(l+1).x, [], 4) ; % Max along orientations
        if isTraining
          res{n}(l+1).aux = bsxfun(@eq, res{n}(l), res{n}(l+1)) ; % Save indicator matrix of where the max values were found (for the backward pass)
          res{n}(l+1).aux = bsxfun(@rdivide, res{n}(l+1).aux, sum(res{n}(l+1).aux, 4)) ; % In case there are several values equal to the max, normalize the values in aux to sum to 1 (along the orientation dimension)
          res{n}(l+1).aux = reshape(res{n}(l+1).aux, sz) ;
        end
        res{n}(l+1).x = reshape(res{n}(l+1).x, sz(1), sz(2), [], sz(4)) ; % Collapse the (now unused, ==1) dimension of orientation groups of the result
        res{n}(l).x = reshape(res{n}(l).x, sz(1), sz(2), sz(3), sz(4)) ; % Restore the dimensions of the input data (res{n}(l).x) as they were before
      case {'conv', 'sharedconv'}
        if strcmp(ly.type, 'sharedconv')
          src = ly.src ; % Save the net/layer IDs where to put the gradient of the shared filters
          ly = net{ly.src(1)}{ly.src(2)} ; % Replace 'ly' by the layer with the original copy of the (shared) filters
        else
          src = [n, l] ; % "Normal case" (no shared filters)
        end

        if ndims(ly.filters) == 5 % Different filters for each recurrent iteration
          nRecurrentIterations = size(ly.filters, 5) ;
          ly.filters = ly.filters(:, :, :, :, recurrentIteration) ; % Select the one of the current iteration
          ly.biases = ly.biases(:, :, recurrentIteration) ; % Select the one of the current iteration
        end

        assert(size(res{n}(l).x, 3) == size(ly.filters, 3)) ; % Check there are no filter groups (not often used, usually a mistake !)
        res{n}(l+1).x = vl_nnconv(res{n}(l).x, ly.filters, ly.biases, 'pad', ly.pad, 'stride', ly.stride) ; % Perform the convolution

      case 'padborders'
        res{n}(l+1).x = padarray(res{n}(l).x, [ly.w, ly.w], 'replicate') ;
        %res{n}(l+1).x = padarray(res{n}(l).x, [ly.w, ly.w], 'symmetric') ;
        %res{n}(l+1).x = padarray(res{n}(l).x, [ly.w, ly.w], 'circular') ;
      case 'downsize'
        res{n}(l+1).x = cnn_downsize(res{n}(l).x, ly.scaleFactor, []) ;
      case 'upsize'
        ly.newSize = round(res{1}(2).originalInputDimensions * ly.newFactorOfOriginalInputDimensions) ;
        %tmp = (0.5 * ((ly.newSize(1) / size(res{n}(l).x, 1)) + (ly.newSize(2) / size(res{n}(l).x, 2)))) / ly.scaleFactor ; % Sanity check of scaleFactor/newSize
        %assert((tmp > 0.75) && (tmp < 1.1)) ; % Sanity check
        assert(ly.scaleFactor >= 1) ;
        res{n}(l+1).x = cnn_upsize(res{n}(l).x, ly.newSize, ly.scaleFactor, ly.scaleGradient, []) ;
      case 'stdnormalization'
        sz = size(res{n}(l).x) ;
        if numel(sz) < 4, sz(4) = 1 ; end % In case there is only 1 image (batch size == 1)
        res{n}(l+1).x = bsxfun(@minus, res{n}(l).x, mean(mean(res{n}(l).x, 1), 2)) ; % Substract the average value of each frame
        res{n}(l+1).x = reshape(res{n}(l+1).x, sz(1), sz(2), sz(3)*sz(4)) ; % Collapse the 3rd/4th dimensions to process all channels images at once
        if ~isfield(net{n}{l}, 'windowMask') || isempty(net{n}{l}.windowMask)
          %net{n}{l}.windowMask = ones(ly.windowSize, 'like', res{n}(l).x) ;
          net{n}{l}.windowMask = logical(fspecial('disk', ceil(ly.windowSize/2))) ;
        end
        stdMap = stdfilt(res{n}(l+1).x, net{n}{l}.windowMask) ;
        e = 0.001 ; % Avoid divisions by zero on the next line
        res{n}(l+1).x = reshape(res{n}(l+1).x ./ (e+stdMap), size(res{n}(l).x)) ; % Divide by std and restore the last 3rd/4th dimensions
        res{n}(l+1).x = bsxfun(@rdivide, res{n}(l+1).x, max(max(abs(res{n}(l+1).x), 1), 2)) ; % Scale values to have max abs value at -1 or +1
      case 'warp'
        isUvMapDeeper = (ly.uvMapSrc(1) > n) || ((ly.uvMapSrc(1) == n) && (ly.uvMapSrc(2) > l)) ; % Logical value: true if we have to fetch UV map for deeper layer: not available at the first recurrent iteration
        if (recurrentIteration == 1) && isUvMapDeeper % No UV map available (yet) ; can happen when doing multiple iterations (for the first one)
          res{n}(l+1).x = res{n}(l).x ; % Copy the data, unchanged
        else
          uvMap = res{ly.uvMapSrc(1)}(ly.uvMapSrc(2)+1).x ;
          %uvMap = res{1}(1).labels ; % For debug only !!
          assert(ly.uvMapUpscalingFactor >= 1) ;
          if ly.uvMapUpscalingFactor > 1
            uvMap = cnn_upsize(uvMap, [size(res{n}(l).x, 1) size(res{n}(l).x, 2)], ly.uvMapUpscalingFactor, 0, []) ; % Upsize UV map
          end
          res{n}(l+1).x = cnn_warp(res{n}(l).x, uvMap, p.referenceFrame, 2, []) ;
          clear uvMap ;
        end
      case 'scale'
        res{n}(l+1).x = res{n}(l).x * ly.s ;
      case 'clearborder'
        r = ly.width ;
        res{n}(l+1).x = res{n}(l).x ;
        res{n}(l+1).x(1:r, :, :, :) = ly.newValue ;
        res{n}(l+1).x(:, 1:r, :, :) = ly.newValue ;
        res{n}(l+1).x(end-r+1:end, :, :, :) = ly.newValue ;
        res{n}(l+1).x(:, end-r+1:end, :, :) = ly.newValue ;
      case 'pool'
        res{n}(l+1).x = vl_nnpool(res{n}(l).x, ly.pool, 'pad', ly.pad, 'stride', ly.stride, 'method', ly.method) ;
      case 'normalizel1'
        res{n}(l+1).x = cnn_normalizel1(res{n}(l).x, ly.s, ly.nGroups, ly.biases, ly.groupDim, []) ;
        if ~isfield(net{n}{l}, 'filters') % Not really needed but create the fields so that we can use the same code as for 'conv' layers
          net{n}{l}.filters = [] ;
          net{n}{l}.filtersLearningRate = 0 ;
          net{n}{l}.filtersWeightDecay = 0 ;
        end
      case 'normalizel2'
        res{n}(l+1).x = cnn_normalizel2(res{n}(l).x, ly.nGroups, ly.groupDim, ly.e, []) ;
      case 'softmax'
        %res{n}(l+1).x = cnn_softmax(res{n}(l).x, []) ;
        if ~isfield(ly, 'nGroups'), net{n}{l}.nGroups = 1 ; end
        if ~isfield(ly, 'groupDim'), net{n}{l}.groupDim = 3 ; end
        res{n}(l+1).x = cnn_softmax(res{n}(l).x, net{n}{l}.nGroups, net{n}{l}.groupDim, []) ;
        if isfield(ly, 's'), res{n}(l+1).x = ly.s * res{n}(l+1).x ; end
      case 'hardmax'
        res{n}(l+1).x = one * bsxfun(@eq, res{n}(l).x, max(res{n}(l).x, [], 3)) ;
      case 'flowloss'
        outputValues = reshape(net{ly.outputValuesSrc(1)}{ly.outputValuesSrc(2)}.filters, p.nSpeedsDecoding*p.nOris, 2) ;
        [loss, res{n}(l+1).aux1, res{n}(l+1).aux2] = cnn_flowloss(res{ly.src(1)}(ly.src(2)+1).x, res{1}(1).labels, outputValues, []) ; % Return the loss at the last layer
        if recurrentIteration == 1 % First iteration
          res{n}(l+1).x = loss ;
        else
          res{n}(l+1).x = res{n}(l+1).x + loss ; % Sum losses over recurrent iterations
        end
      case 'epeloss'
        res{n}(l+1).x = cnn_epeloss(res{n}(l).x, res{1}(1).labels, []) ;
      case 'aaeloss'
        res{n}(l+1).x = cnn_aaeloss(res{n}(l).x, res{1}(1).labels, []) ;
      case 'relu'
        res{n}(l+1).x = max(res{n}(l).x, single(0)) ;
      case 'square'
        res{n}(l+1).x = res{n}(l).x .^ 2 ;
      case 'exp'
        res{n}(l+1).x = exp(res{n}(l).x) ;
      case 'tanh'
        res{n}(l+1).x = tanh(res{n}(l).x) ;
      case 'logistic'
        res{n}(l+1).x = 1 ./ (1 + exp(-res{n}(l).x)) ;
      case 'noffset'
        res{n}(l+1).x = cnn_noffset(res{n}(l).x, ly.param) ;
      case 'dropout'
        if ~isTraining || p.overfit
          res{n}(l+1).x = res{n}(l).x ; % Disable dropout
        else
          [res{n}(l+1).x, res{n}(l+1).aux] = cnn_dropout(res{n}(l).x, 'rate', ly.rate) ; % Enable dropout
        end
      case 'concat'
        res{n}(l+1).x = cat(3, res{ly.src(1)}(ly.src(2)+1).x, res{n}(l).x) ;
        net{n}{l}.n1 = size(res{ly.src(1)}(ly.src(2)+1).x, 3) ; % Save the number of channels of the source
        if p.conserveMemory, res{ly.src(1)}(ly.src(2)+1).x = [] ; end % Clear the source: assume it is not used several times !
      case 'ori-concat'
        % Concatenate channels, but do not mix up orientation groups (keep orientation groups as the last/outermost dimension)
        sz = [ size(res{n}(l).x, 1), size(res{n}(l).x, 2), size(res{n}(l).x, 3), size(res{n}(l).x, 4) ] ;
        res{n}(l+1).x = cat(3, ... % Expand orientation groups of channels, then concat along the other channels (keeping orientation groups)
          reshape(res{ly.src(1)}(ly.src(2)+1).x, sz(1), sz(2), [], ly.nOris, sz(4)), ...
          reshape(res{n}(l).x, sz(1), sz(2), [], ly.nOris, sz(4)) ) ;
        res{n}(l+1).x = reshape(res{n}(l+1).x, sz(1), sz(2), [], sz(4)) ; % Collapse the orientation groups with the other channels
        net{n}{l}.n1 = size(res{ly.src(1)}(ly.src(2)+1).x, 3) / ly.nOris ; % Save the number of channels of the source
        if p.conserveMemory, res{ly.src(1)}(ly.src(2)+1).x = [] ; end % Clear the source: assume it is not used several times !
      case 'flowRecurrent'
        if ly.sumResults && (recurrentIteration > 1)
          res{n}(l+1).x = res{n}(l+1).x + res{n}(l).x ; % Sum new values with the results of the previous iteration
        else
          res{n}(l+1).x = res{n}(l).x ; % Place with new values
        end
        if recurrentIteration < net{n}{l}.nRecurrentIterations
          backTo = ly.backTo ; % Go back to the specified subnet/layer
        else
          backTo = [1 1] ; % Will stop the recurrent iterations
        end
      otherwise
        error('Unknown layer type %s', ly.type) ;
    end % switch ly.type
    clear ly ;

    % Erase the data at the previous layer
    %if (~isTraining || p.conserveMemory) && ly.canErase
    if p.conserveMemory && (l > 1) && net{n}{l-1}.canErase && ~(reuseIntermediateResults && ((l-1) == net{n}{1}.earlyStopMaxLayer))
      res{n}(l).x = [] ; % Clear the intermediate result
    end

    if p.sync && p.useGpu
      wait(gpuDevice) ; % This should make things slower, but on MATLAB 2014a it is necessary for any decent performance.
    end
    res{n}(l).time = res{n}(l).time + toc(timer) ;

    % Iterate
    if (l == numel(net{n})) && (n == numel(net)) % Reached the last layer of the last subnet
      break ;
    elseif l == numel(net{n}) % Reached the last layer of a subnet (not the last one)
      n = n + 1 ; l = 1 ; % Move to the first layer of the next subnet
    else
      l = l + 1 ; % Next layer, same subnet
    end
  end % For each subnet/layer

  if isTraining
    %----------------------------------------------------------------------
    % Backward
    %----------------------------------------------------------------------
    res{n}(l+1).dzdx = one ; % Initialize the derivative at the last output of the last subnet

    while true % For each subnetwork/layer
      %disp([n, l]) ; % Debug display

      ly = net{n}{l} ;

      % Do an "early stop" (do not backpropagate to the very beginning of the network) if we encounter a learning rate = 0
      if l <= net{n}{1}.earlyStopMaxLayer
        if p.conserveMemory, res{n}(l+1).dzdx = [] ; end
        for l2 = 1:l, res{n}(l2).backwardTime = 0 ; end % Set 0 time for the remaining (skipped) layers
        if n == 1  % We are in the first subnet
          break ; % Stop the backward evaluation
        else
          n = n - 1 ; l = length(net{n}) ; % Move to the previous subnet, last layer
          continue ;
        end
      end

      timer = tic() ;

      %[n, l], keyboard % Debug

      if ~isempty(res{n}(l+1).dzdx) % Can happen with 'copy' layers that effectively make the forward evaluation 'skip' layers (by copying the value of a few layers back)
        switch ly.type
          case 'getBatch' % Nothing to do
          case 'identity'
            res{n}(l).dzdx = res{n}(l+1).dzdx ;
          case 'copy'
            res{ly.src(1)}(ly.src(2)+1).dzdx = res{n}(l+1).dzdx ;
          case 'substractconv'
            error('Not supported !') ;
          case 'ori-conv'
            res{n}(l).x(isnan(res{n}(l).x)) = 0 ; % Replace NaNs (pixels with unknown ground truth) by 0s (null derivative)
            [res{n}(l).dzdx, dzdwTmp{1}, dzdwTmp{2}] = vl_nnconv(res{n}(l).x, ly.allFilters, ly.allBiases, res{n}(l+1).dzdx, 'pad', ly.pad, 'stride', ly.stride) ;

            if (ly.filtersLearningRate > 0) || (ly.biasesLearningRate > 0)
              res{n}(l).nShared = 1 ;
              % Gradient wrt the filters
              if p.useGpu && issparse(ly.weightInverseProjection) && verLessThan('matlab', '8.5') % Sparse arrays on GPU not supported in Matlab pre-R2015a
                dzdwTmp{1} = ly.weightInverseProjection * double(gather(dzdwTmp{1}(:))) ; % Transfer to CPU then run sparse multiplication
                dzdwTmp{1} = gpuArray(single(dzdwTmp{1})) ; % Transfer result back to GPU
              else
                dzdwTmp{1} = ly.weightInverseProjection * double(dzdwTmp{1}(:)) ; % ly.weightInverseProjection is sparse and therefore double (Matlab does not support sparse singles)
                dzdwTmp{1} = single(dzdwTmp{1}) ;
              end
              res{n}(l).dzdw{1} = reshape(dzdwTmp{1}, size(ly.filters)) ; % Save the gradient
              % Gradient wrt the biases
              dzdwTmp{2} = mean(reshape(dzdwTmp{2}, 1, [], ly.nOrientations), 3) ; % Average across orientations
              res{n}(l).dzdw{2} = reshape(dzdwTmp{2}, size(ly.biases)) ; % Save the gradient
            end
          case 'ori-maxpool'
            res{n}(l).dzdx = res{n}(l+1).dzdx .* res{n}(l+1).aux ;
            keyboard % TODO CHECK
          case {'conv', 'sharedconv'}
            if strcmp(ly.type, 'sharedconv')
              src = ly.src ; % Save the net/layer IDs where to put the gradient of the shared filters
              ly = net{ly.src(1)}{ly.src(2)} ; % Replace 'ly' by the layer with the original copy of the (shared) filters
            else
              src = [n, l] ; % "Normal case" (no shared filters)
            end

            if ndims(ly.filters) == 5 % Different filters for each recurrent iteration
              % Select filters/biases of the current iteration
              ly.filters = ly.filters(:, :, :, :, recurrentIteration) ;
              ly.biases = ly.biases(:, :, recurrentIteration) ;
            end

            res{n}(l).x(isnan(res{n}(l).x)) = 0 ; % Replace NaNs (pixels with unknown ground truth) by 0s (null derivative)
            [res{n}(l).dzdx, dzdwTmp{1}, dzdwTmp{2}] = vl_nnconv(res{n}(l).x, ly.filters, ly.biases, res{n}(l+1).dzdx, 'pad', ly.pad, 'stride', ly.stride) ;

            if (ly.filtersLearningRate > 0) || (ly.biasesLearningRate > 0)
              % Save the gradient (accumulate if weights are shared by several layers)
              if isempty(res{src(1)}(src(2)).nShared)
                % Empty initialization
                res{src(1)}(src(2)).dzdw{1} = zeros(size(net{src(1)}{src(2)}.filters), 'like', net{src(1)}{src(2)}.filters) ;
                res{src(1)}(src(2)).dzdw{2} = zeros(size(net{src(1)}{src(2)}.biases), 'like', net{src(1)}{src(2)}.biases) ;
                res{src(1)}(src(2)).nShared = 0 ; % Number of times the weights are used (1 normally, > 1 if shared by several layers)
              end
              if differentFiltersPerIteration
                % Different filters for each recurrent iteration
                res{src(1)}(src(2)).dzdw{1}(:, :, :, :, recurrentIteration) = dzdwTmp{1} ;
                res{src(1)}(src(2)).dzdw{2}(:, :, recurrentIteration) = dzdwTmp{2} ;
                res{src(1)}(src(2)).nShared = 1 ;
              else
                % Normal case: accumulate (sum) gradients between shared layers
                res{src(1)}(src(2)).dzdw{1} = res{src(1)}(src(2)).dzdw{1} + dzdwTmp{1} ;
                res{src(1)}(src(2)).dzdw{2} = res{src(1)}(src(2)).dzdw{2} + dzdwTmp{2} ;
                res{src(1)}(src(2)).nShared = res{src(1)}(src(2)).nShared + 1; % Increment
              end
            end

          case 'padborders'
            res{n}(l).dzdx = res{n}(l+1).dzdx(ly.w+1 : end-ly.w, ly.w+1 : end-ly.w, :, :) ; % Center crop
          case 'downsize'
            res{n}(l).dzdx = cnn_downsize(res{n}(l).x, ly.scaleFactor, res{n}(l+1).dzdx) ;
          case 'upsize'
            ly.newSize = round(res{1}(2).originalInputDimensions * ly.newFactorOfOriginalInputDimensions) ;
            res{n}(l).dzdx = cnn_upsize(res{n}(l).x, ly.newSize, ly.scaleFactor, ly.scaleGradient, res{n}(l+1).dzdx) ;
          case 'stdnormalization'
            error('Not supported !') ;
          case 'warp'
            error('Not supported !') ;
          case 'reshape'
            res{n}(l).dzdx = reshape(res{n}(l+1).dzdx, size(res{n}(l).x)) ;
          case 'add'
            res{n}(l).dzdx = res{n}(l+1).dzdx ;
          case 'scale'
            res{n}(l).dzdx = ly.s * res{n}(l+1).dzdx ;
          case 'clearborder'
            r = ly.width ;
            res{n}(l).dzdx = res{n}(l+1).dzdx ;
            res{n}(l).dzdx(1:r, :, :, :) = single(0) ;
            res{n}(l).dzdx(:, 1:r, :, :) = single(0) ;
            res{n}(l).dzdx(end-r+1:end, :, :, :) = single(0) ;
            res{n}(l).dzdx(:, end-r+1:end, :, :) = single(0) ;
          case 'pool'
            res{n}(l).dzdx = vl_nnpool(res{n}(l).x, ly.pool, res{n}(l+1).dzdx, 'pad', ly.pad, 'stride', ly.stride, 'method', ly.method) ;
          case 'normalizel1'
            if ly.biasesLearningRate == 0
              res{n}(l).dzdx = cnn_normalizel1(res{n}(l).x, ly.s, ly.nGroups, ly.biases, ly.groupDim, res{n}(l+1).dzdx) ; % No need to compute dzdw
            else
              [res{n}(l).dzdx, dzdwTmp{2}] = cnn_normalizel1(res{n}(l).x, ly.s, ly.nGroups, ly.biases, ly.groupDim, res{n}(l+1).dzdx) ;
              % Save the gradient
              if isempty(res{n}(l).nShared)
                res{n}(l).dzdw{2} = dzdwTmp{2} ;
                res{n}(l).nShared = 1 ;
              else
                res{n}(l).dzdw{2} = res{n}(l).dzdw{2} + dzdwTmp{2} ; % Accumulate (sum) gradients between shared layers
                res{n}(l).nShared = res{n}(l).nShared + 1; % Increment
              end
            end
          case 'normalizel2'
            res{n}(l).dzdx = cnn_normalizel2(res{n}(l).x, ly.nGroups, ly.groupDim, ly.e, res{n}(l+1).dzdx) ; % No need to compute dzdw
          case 'softmax'
            res{n}(l).dzdx = cnn_softmax(res{n}(l).x, ly.nGroups, ly.groupDim, res{n}(l+1).dzdx) ;
            if isfield(ly, 's'), res{n}(l).dzdx = ly.s * res{n}(l+1).dzdx ; end
          case 'hardmax'
            res{n}(l).dzdx = bsxfun(@eq, res{n}(l).x, max(res{n}(l).x, [], 3)) .* res{n}(l+1).dzdx ;
          case 'flowloss'
            outputValues = reshape(net{ly.outputValuesSrc(1)}{ly.outputValuesSrc(2)}.filters, p.nSpeedsDecoding*p.nOris, 2) ;
            res{ly.src(1)}(ly.src(2)+1).dzdx = cnn_flowloss(res{ly.src(1)}(ly.src(2)+1).x, res{1}(1).labels, outputValues, res{n}(l+1).dzdx, res{n}(l+1).aux1, res{n}(l+1).aux2) ;
            n = ly.src(1) ; l = ly.src(2)+1 ; % Go to earlier layer (skip some)
          case 'epeloss'
            res{n}(l).dzdx = cnn_epeloss(res{n}(l).x, res{1}(1).labels, res{n}(l+1).dzdx) ;
          case 'aaeloss'
            res{n}(l).dzdx = cnn_aaeloss(res{n}(l).x, res{1}(1).labels, res{n}(l+1).dzdx) ;
          case 'relu'
            res{n}(l).dzdx = (res{n}(l).x > single(0)) .* res{n}(l+1).dzdx ;
          case 'square'
            %res{n}(l).dzdx = 2 * res{n}(l).x .* res{n}(l+1).dzdx ;
            res{n}(l).dzdx = res{n}(l).x .* res{n}(l+1).dzdx ; % (Surprisingly) Much faster
          case 'exp'
            res{n}(l).dzdx = exp(res{n}(l).x) .* res{n}(l+1).dzdx ;
          case 'tanh'
            res{n}(l).dzdx = (1 - tanh(res{n}(l).x).^2) .* res{n}(l+1).dzdx ;
          case 'logistic'
            tmp = 1 ./ (1 + exp(-res{n}(l).x)) ;
            res{n}(l).dzdx = tmp .* (1 - tmp) .* res{n}(l+1).dzdx ;
          case 'noffset'
            res{n}(l).dzdx = cnn_noffset(res{n}(l).x, ly.param, res{n}(l+1).dzdx) ;
          case 'dropout'
            if ~isTraining || p.overfit
              res{n}(l).dzdx = res{n}(l+1).dzdx ; % Disable dropout
            else
              res{n}(l).dzdx = cnn_dropout(res{n}(l).x, res{n}(l+1).dzdx, 'mask', res{n}(l+1).aux) ; % Enable dropout
            end
          case 'concat'
            res{n}(l).dzdx                   = res{n}(l+1).dzdx(:, :, ly.n1+1 : end, :) ; % Keep the relevant (first) channels of dzdx
            res{ly.src(1)}(ly.src(2)+1).dzdx = res{n}(l+1).dzdx(:, :, 1 : ly.n1, :) ; % Put the other channels where we sourced the concatenated data
          case 'ori-concat'
            sz = [ size(res{n}(l+1).dzdx, 1), size(res{n}(l+1).dzdx, 2), size(res{n}(l+1).dzdx, 3), size(res{n}(l+1).dzdx, 4) ] ;
            res{n}(l).dzdx = reshape(res{n}(l+1).dzdx, sz(1), sz(2), sz(3)/ly.nOris, ly.nOris, sz(4)) ; % Expand the orientation groups
            res{ly.src(1)}(ly.src(2)+1).dzdx = res{n}(l).dzdx(:, :, 1 : ly.n1, :, :) ; % Put the other channels where we sourced the concatenated data
            res{n}(l).dzdx                   = res{n}(l).dzdx(:, :, ly.n1+1 : end, :, :) ; % Keep the relevant (first) channels of dzdx
            % Collapse the orientation groups with the other channels
            res{ly.src(1)}(ly.src(2)+1).dzdx = reshape(res{ly.src(1)}(ly.src(2)+1).dzdx, sz(1), sz(2), [], sz(4)) ;
            res{n}(l).dzdx                   = reshape(res{n}(l).dzdx                  , sz(1), sz(2), [], sz(4)) ;
          case 'flowRecurrent'
            res{n}(l).dzdx = res{n}(l+1).dzdx ;
          otherwise
            error('Unknown layer type %s', ly.type) ;
        end % switch ly.type
      end % if ~isempty(res{n}(l+1).dzdx)

      if p.conserveMemory, res{n}(l+1).dzdx = [] ; end
      if p.sync && p.useGpu
        wait(gpuDevice) ;
      end
      res{n}(l).backwardTime = res{n}(l).backwardTime + toc(timer) ;

      % Iterate
      if (n == 1) && (l == 1)  % We reached the first subnet/layer
        if p.conserveMemory, res{n}(l).dzdx = [] ; end
        break ; % Stop the backward evaluation
      elseif l == 1
        if p.conserveMemory, res{n}(l).dzdx = [] ; end
        n = n - 1 ; l = length(net{n}) ; % Move to the previous subnet, last layer
      else
        l = l - 1 ; % Same subnet, previous layer
      end
    end % For each subnet/layer
  end % if isTraining

  if ~isequal(backTo, [1 1])
    n = backTo(1) ; l = backTo(2) ; % Recurrent connection: start another forward evaluation from there
  else
    return; % No recurrent connections: stop here
  end
end % Loop for recurrent connections
