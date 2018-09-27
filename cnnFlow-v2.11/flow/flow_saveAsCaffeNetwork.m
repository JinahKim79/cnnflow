function flow_saveAsCaffeNetwork(p, net, inputSz, networkName, testCaffeModel)
%FLOW_SAVEASCAFFENETWORK Save CNN network as a Caffe model (both .prototxt network definition file and .caffemodel binary weights).
% Requires Caffe built with the MatCaffe interface.

% Author: Zohar Bar-Yehuda, Damien Teney

if (nargin < 3) || isempty(networkName) % Missing argument
  networkName = 'net' ; % Set default value
end

if testCaffeModel
  % Load, as test input, the first element of the dataset used to train the network
  tmp1 = imread([p.dataDir, filesep, p.files.im{1}{1}, '.png']) ;
  tmp2 = imread([p.dataDir, filesep, p.files.im{1}{2}, '.png']) ;
  tmp3 = imread([p.dataDir, filesep, p.files.im{1}{3}, '.png']) ;
  inputData = cat(3, rgb2gray(tmp1), rgb2gray(tmp2), rgb2gray(tmp3)) ;
  inputData = single(inputData) ./ 255 ;
  if isempty(inputSz)
    inputSz = size(inputData) ; % Automatic input size
  else
    assert(length(inputSz) == 2) ;
    inputSz(3) = p.nFrames ;
    inputData = crop(inputData, inputSz(1), inputSz(2), 'center') ; % Crop to given size
  end
else
  % Given input size
  assert(length(inputSz) == 2) ;
  inputSz(3) = p.nFrames ;
  inputData = zeros(inputSz(1), inputSz(2), inputSz(3), 1, 'single') ; % Create dummy data
end

% Run data through the network
p2 = p ;
p2.conserveMemory = false ; % Make sure we keep all intermediate results
p2.useGpu = false ; % Run on CPU to make sure there is enough memory for p.conserveMemory = false
net = moveCnnGpu(p2, net) ; % Make sure it's on the CPU/GPU as needed now (may have been trained differently)
assert(strcmp(net{1}{1}.type, 'getBatch')) ; % Make sure the first layer is a 'getBatch' layer

% Replace the getBatch layer by a custom function to force-feed our test data
function [images, targets, originalInputDimensions] = getBatchDummy(p, ~, ~)
  images = inputData ;
  if p.useGpu
    images = gpuArray(images) ;
  end
  targets = zeros(size(inputData, 1), size(inputData, 2), 2, 1, 'like', images) ;
  if p.stride(end) > 1, targets = targets(1:p.stride(end):end, 1:p.stride(end):end, :, :) ; end
  originalInputDimensions = [size(inputData, 1), size(inputData, 2)] ;
  assert(p.stride(1) == 1) ;
end
net{1}{1}.getBatchFcn = @getBatchDummy ;

fprintf('Evaluating the Matlab network...\n') ;
[res, net] = evalCnn(p2, net, [], {}, false, false) ; % Evaluate the CNN (forward)

%--------------------------------------------------------------------------
% Write the prototxt file
%--------------------------------------------------------------------------
prototxtFilename = fullfile(p.expDir, [networkName, '.prototxt']) ;
fprintf('Writing the prototxt file...\n\t%s\n', prototxtFilename) ;
modelFilename = fullfile(p.expDir, [networkName, '.caffemodel']) ;
fileId = fopen(prototxtFilename, 'w') ;
if fileId == -1
  error(['Cannot create file: ', prototxtFilename]) ;
end
fprintf(fileId, 'name: "%s"\n\n', networkName) ;

% Input dimensions
fprintf(fileId, 'input: "data"\n') ;
fprintf(fileId, 'input_dim: 1\n') ;
fprintf(fileId, 'input_dim: %d\n', inputSz(3)) ;
fprintf(fileId, 'input_dim: %d\n', inputSz(1)) ;
fprintf(fileId, 'input_dim: %d\n\n', inputSz(2)) ;

decodingStage = false ;

function name = makeLayerName(n, caffeLayerId, decodingStage)
  if decodingStage
    name = sprintf('decoding-layer%d', caffeLayerId) ; % Automatic layer name
  else
    name = sprintf('scale%d-layer%d', n, caffeLayerId) ; % Automatic layer name
  end
end

for n = 1:length(net) % For each subnet
  caffeLayerId = 1 ;
  for l = 1:length(net{n}) % For each layer
    if (n == length(net)) && (l > net{end}{1}.outputLayer) % We just processed the layer marked as the output layer (the one before the loss)
      break ; % Stop here
    end

    data = res{n}(l).x ; % Data given as input to the current layer (rename for readability)
    nChannels = size(data, 3) ;

    % Write the current layer
    switch lower(net{n}{l}.type)
      %--------------------------------------------------------------------
      case 'getbatch'
        net{n}{l}.name = 'data' ;
        continue ;
      case 'copy'
        net{n}{l}.name = net{net{n}{l}.src(1)}{net{n}{l}.src(2)}.name ;
        continue ;
      case {'identity', 'dropout', 'stdnormalization', 'warp', 'clearborder', 'flowrecurrent'} % Unsupported layers
        net{n}{l}.name = net{n}{l-1}.name ;
        continue ;
      %--------------------------------------------------------------------
      case 'downsize'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        scaleFactor = round(1 / net{n}{l}.scaleFactor) ;
        assert(scaleFactor >= 1) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Convolution"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;
        fprintf(fileId, '  convolution_param {\n') ;
        fprintf(fileId, '    kernel_size: %d\n', 2*scaleFactor - mod(scaleFactor, 2)) ;
        fprintf(fileId, '    stride: %d\n', scaleFactor) ;
        fprintf(fileId, '    num_output: %d\n', nChannels) ;
        fprintf(fileId, '    group: %d\n', nChannels) ;
        fprintf(fileId, '    pad: %d\n', ceil((scaleFactor - 1) / 2) ) ;
        fprintf(fileId, '    weight_filler: { type: "bilinear" }\n') ;
        fprintf(fileId, '    bias_term: false\n') ;
        fprintf(fileId, '  }\n') ;
        fprintf(fileId, '  param { lr_mult: 0 decay_mult: 0 }\n') ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'upsize'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        scaleFactor = net{n}{l}.scaleFactor ;
        if scaleFactor == 1 % No action: skip
          %net{n}{l}.name = net{n}{l-1}.name ;
          %continue ;
        end
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Deconvolution"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;
        fprintf(fileId, '  convolution_param {\n') ;
        fprintf(fileId, '    kernel_size: %d\n', 2*scaleFactor - mod(scaleFactor, 2)) ;
        fprintf(fileId, '    stride: %d\n', scaleFactor) ;
        fprintf(fileId, '    num_output: %d\n', nChannels) ;
        fprintf(fileId, '    group: %d\n', nChannels) ;
        fprintf(fileId, '    pad: %d\n', ceil((scaleFactor - 1) / 2) ) ;
        fprintf(fileId, '    weight_filler: { type: "bilinear" }\n') ;
        fprintf(fileId, '    bias_term: false\n') ;
        fprintf(fileId, '  }\n') ;
        fprintf(fileId, '  param { lr_mult: 0 decay_mult: 0 }\n') ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case {'conv', 'ori-conv', 'substractconv'}
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        if strcmp(net{n}{l}.type, 'ori-conv')
          % Replace filters/biases by .allFilters/.allBiases
          net{n}{l}.filters = net{n}{l}.allFilters ;
          net{n}{l}.biases = net{n}{l}.allBiases ;
        elseif strcmp(net{n}{l}.type, 'substractconv')
          % Turn filters into substraction filters
          filters = net{n}{l}.filters;
          center = floor(1 + (size(filters) - 1) / 2) ;
          for i = 1:size(filters, 3)
            for j = 1:size(filters, 4)
              filters(:, :, i, j) = -filters(:, :, i, j) ;
            end
            filters(center(1), center(2), i, i) = filters(center(1), center(2), i, i) + 1 ;
          end
          assert(size(filters, 3) == size(filters, 4)) ;
          net{n}{l}.filters = filters ; % Save the modified filters
          clear filters ;
        end

        % Rename for readability
        filters = net{n}{l}.filters;

        %if size(filters, 1) > 1 || size(filters, 2) > 1 % TODO: check carefully for fully-connected layers (compare filter input size and number of elements in the feature map)
        if true
          % Convolution layer
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "Convolution"\n') ;
          fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
          writeBlobNames(fileId, net, n, l) ;
          fprintf(fileId, '  convolution_param {\n') ;
          writeKernelSize(fileId, [size(filters, 1), size(filters, 2)]) ;
          fprintf(fileId, '    num_output: %d\n', size(filters, 4)) ;
          writeStride(fileId, net{n}{l}) ;
          if isfield(net{n}{l}, 'pad') && length(net{n}{l}.pad) == 4
            if net{n}{l}.pad(1) ~= net{n}{l}.pad(2) || net{n}{l}.pad(3) ~= net{n}{l}.pad(4)
              error('Caffe only supports symmetrical padding') ;
            end
          end
          writePad(fileId, net{n}{l}) ;
          nGroups = size(data, 3) / size(filters, 3) ;
          assert(mod(nGroups, 1) == 0) ;
          if nGroups > 1
            fprintf(fileId, '    group: %d\n', nGroups) ;
          end
          fprintf(fileId, '  }\n') ;
          fprintf(fileId, '}\n\n') ;
        else
          % Fully connected layer
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "InnerProduct"\n') ;
          fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
          writeBlobNames(fileId, net, n, l) ;
          fprintf(fileId, '  inner_product_param {\n') ;        
          fprintf(fileId, '  num_output: %d\n', size(filters, 4)) ;        
          fprintf(fileId, '  }\n') ;
          fprintf(fileId, '}\n\n') ;
        end
      %--------------------------------------------------------------------
      case 'relu'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "ReLU"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'pool'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Pooling"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        % Check padding compatability with Caffe (http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf)
        if ~isfield(net{n}{l}, 'pad')
          net{n}{l}.pad = [0 0 0 0];
        elseif length(net{n}{l}.pad) == 1
          net{n}{l}.pad = repmat(net{n}{l}.pad, 1, 4) ;
        end
        if ~isfield(net{n}{l}, 'stride')
          net{n}{l}.stride = [1 1];
        elseif length(net{n}{l}.stride) == 1
          net{n}{l}.stride = repmat(net{n}{l}.stride, 1, 2) ;
        end
        if length(net{n}{l}.pool) == 1
          net{n}{l}.pool = repmat(net{n}{l}.pool, 1, 2) ;
        end

        %{
        support = net{n}{l}.pool;
        stride = net{n}{l}.stride;
        pad = net{n}{l}.pad;
        compatability_pad_y = ceil((size(data, 1)-support(1)) / stride(1)) * stride(1) + support(1) - size(data, 1) ;
        compatability_pad_x = ceil((size(data, 2)-support(2)) / stride(2)) * stride(2) + support(2) - size(data, 2) ;
        if (pad(2) ~= pad(1) + compatability_pad_y) || ...
           (pad(4) ~= pad(3) + compatability_pad_x)
          displayWarning('Changing padding in pooling layer net{%d}{%d} for compatibility with Caffe: [%d %d %d %d]\n', n, l, pad(1), pad(1)+compatability_pad_y, pad(3) , pad(3)+compatability_pad_x) ;
          net{n}{l}.pad = [pad(1), pad(1)+compatability_pad_y, pad(3) , pad(3)+compatability_pad_x] ; % Change the padding
        end
        %}
        net{n}{l}.pad(:) = 0 ;
%net{n}{l}.pad(:) = 1 ;

        writeBlobNames(fileId, net, n, l) ;
        fprintf(fileId, '  pooling_param {\n') ; 
        switch net{n}{l}.method
          case 'max', fprintf(fileId, '    pool: MAX\n') ;
          case 'avg', fprintf(fileId, '    pool: AVE\n') ;
          otherwise, error('Unknown pooling type') ;
        end
        writeKernelSize(fileId, net{n}{l}.pool) ;
        writeStride(fileId, net{n}{l}) ;
        writePad(fileId, net{n}{l}) ;
        fprintf(fileId, '  }\n') ;                  
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'normalize' 
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        % MATLAB parameters: [local_size, kappa, alpha/local_size, beta]
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "LRN"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;
        fprintf(fileId, '  lrn_param {\n') ; 
        fprintf(fileId, '    local_size: %d\n', net{n}{l}.param(1)) ;
        fprintf(fileId, '    k: %f\n', net{n}{l}.param(2)) ;
        fprintf(fileId, '    alpha: %f\n', net{n}{l}.param(3)*net{n}{l}.param(1)) ;
        fprintf(fileId, '    beta: %f\n', net{n}{l}.param(4)) ;
        fprintf(fileId, '  }\n') ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'softmax'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Softmax"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;  
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'tanh'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "TanH"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;  
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'sigmoid'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Sigmoid"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;  
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'square'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Power"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;  
        fprintf(fileId, '  power_param {\n') ;
        fprintf(fileId, '    power: %f\n', 2) ;
        fprintf(fileId, '    scale: %f\n', 1) ;
        fprintf(fileId, '    shift: %f\n', 0) ;
        fprintf(fileId, '  }\n') ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      case 'scale'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Power"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        writeBlobNames(fileId, net, n, l) ;  
        fprintf(fileId, '  power_param {\n') ;
        fprintf(fileId, '    power: %f\n', 1) ;
        fprintf(fileId, '    scale: %f\n', ly.s) ;
        fprintf(fileId, '    shift: %f\n', 0) ;
        fprintf(fileId, '  }\n') ;
        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      %case 'normalizel1' % TODO (no easy way to do this in Caffe)
      %--------------------------------------------------------------------
      case 'normalizel2'
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;
        if net{n}{l}.nGroups == 1 % Simple case: normalization across ALL channels
          % L2 normalization
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "Normalize"\n') ;
          fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
          writeBlobNames(fileId, net, n, l) ;
          fprintf(fileId, '}\n\n') ;

          % L2 normalization (longer code using LRN, in case Caffe's layer "Normalize" is not available
          %{
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "LRN"\n') ;
          fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
          writeBlobNames(fileId, net, n, l) ;
          fprintf(fileId, '  lrn_param {\n') ; 
          fprintf(fileId, '    norm_region: ACROSS_CHANNELS\n') ;
          fprintf(fileId, '    local_size: %d\n', 2*nChannels+1) ; % Across ALL channels
          fprintf(fileId, '    k: %f\n', net{n}{l}.biases ) ;
          assert(numel(net{n}{l}.biases) == 1) ;
          fprintf(fileId, '    alpha: %f\n', 1.0) ;
          fprintf(fileId, '    beta: %f\n', 0.5) ;
          fprintf(fileId, '  }\n') ;
          fprintf(fileId, '}\n\n') ;
          %}
        else % Normalization across groups of channels: slice channels into intermediate blobs, then normalize all channels of those, then concatenate the results
          groupSize = nChannels / net{n}{l}.nGroups ;

          % Slicer
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "Slice"\n') ;
          fprintf(fileId, '  name: "%s-slicer"\n', net{n}{l}.name) ;
          for k = 1:net{n}{l}.nGroups % For each group of channels
            fprintf(fileId, '  top: "%s-slice%d"\n', net{n}{l}.name, k) ;
          end
          fprintf(fileId, '  bottom: "%s"\n', net{n}{l-1}.name) ;
          fprintf(fileId, '  slice_param {\n') ;
          fprintf(fileId, '    axis: 1\n') ;
          for k = 2:net{n}{l}.nGroups % For each delineation beteen successive groups of channels
            fprintf(fileId, '    slice_point: %d\n', (k-1) * groupSize) ;
          end
          fprintf(fileId, '  }\n') ;
          fprintf(fileId, '}\n') ;

          % Normalization of every slice
          for k = 1:net{n}{l}.nGroups % For each group of channels
            fprintf(fileId, 'layer {\n') ;
            fprintf(fileId, '  name: "%s-slice%dnormalized"\n', net{n}{l}.name, k) ;
            fprintf(fileId, '  top: "%s-slice%dnormalized"\n', net{n}{l}.name, k) ;
            fprintf(fileId, '  bottom: "%s-slice%d"\n', net{n}{l}.name, k) ;
            %fprintf(fileId, '  type: "Normalize"\n') ; % Not good, not elementwise !!!
            %%{
            fprintf(fileId, '  type: "LRN"\n') ;
            fprintf(fileId, '  lrn_param {\n') ; 
            fprintf(fileId, '    norm_region: ACROSS_CHANNELS\n') ;
            %fprintf(fileId, '    local_size: %d\n', 2*groupSize + 1) ;
            fprintf(fileId, '    local_size: %d\n', 2*(groupSize-1)+1) ;
            %fprintf(fileId, '    local_size: %d\n', 1) ;
            fprintf(fileId, '    engine: CAFFE\n') ;
            fprintf(fileId, '    k: %f\n', net{n}{l}.e ) ;
            fprintf(fileId, '    alpha: %f\n', 1.0) ;
            fprintf(fileId, '    beta: %f\n', 0.5) ;
            fprintf(fileId, '  }\n') ;
            %}
            fprintf(fileId, '}\n') ;
          end

          % Concatenate normalized values
          fprintf(fileId, 'layer {\n') ;
          fprintf(fileId, '  type: "Concat"\n') ;
          fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
          fprintf(fileId, '  top: "%s"\n', net{n}{l}.name) ;
          % Write source blobs
          for k = 1:net{n}{l}.nGroups % For each group of channels
            fprintf(fileId, '  bottom: "%s-slice%dnormalized"\n', net{n}{l}.name, k) ;
          end
          fprintf(fileId, '}\n\n') ;

        end % if net{n}{l}.nGroups == 1
      %--------------------------------------------------------------------
      case 'ori-concat'
        if n < length(net) % Not in the last subnet
          continue ; % Skip if not in the last subnet (we perform only 1 concat instead of multiple pairwise concats as in the Matlab version)
        end

        decodingStage = true ; % Will name layers differently from now on
        caffeLayerId = 1 ; % Reset numbering for the layers name
        net{n}{l}.name = makeLayerName(n, caffeLayerId, decodingStage) ;

        fprintf(fileId, 'layer {\n') ;
        fprintf(fileId, '  type: "Concat"\n') ;
        fprintf(fileId, '  name: "%s"\n', net{n}{l}.name) ;
        fprintf(fileId, '  top: "%s"\n', net{n}{l}.name) ;

        % Write source blobs
        for n2 = 1:n % For each source to concatenate
          if n2 < n
            if strcmp(net{n2}{end}.type, 'ori-concat')
              bottomName = net{n2}{end-1}.name ;
            else
              bottomName = net{n2}{end}.name ;
            end
          else % n2 == n
            bottomName = net{n2}{l-1}.name ;
          end
          fprintf(fileId, '  bottom: "%s"\n', bottomName) ;
        end
        % Modify filters of the following convolutional layer
        % Since concatenate with the scale as the outermost dimension (not the orientation as in the Matlab version), we switch those dimensions in the weights of the following layer
        assert(strcmp(net{n}{l+1}.type, 'ori-conv')) ;
        filters = net{n}{l+1}.allFilters ; % Make a copy
        nScales = length(net) ; % Number of concatenated sources
        filters = reshape(filters, net{n}{l+1}.filterSize, net{n}{l+1}.filterSize, [], nScales, net{n}{l+1}.nOrientations, size(filters, 4)) ; % Expand input dimensions for the groups of concatenated channels
        filters = permute(filters, [1 2 3 5 4 6]) ; % Swap 2 dimensions
        filters = reshape(filters, net{n}{l+1}.filterSize, net{n}{l+1}.filterSize, [], size(filters, 6)) ; % Collapse back to 4 dimensions
        net{n}{l+1}.allFilters = filters ; % Save the modified filters

        fprintf(fileId, '}\n\n') ;
      %--------------------------------------------------------------------
      otherwise
        displayWarning('Unsupported layer type\n') ;
        net{n}{l}
        keyboard
      %--------------------------------------------------------------------
    end
    caffeLayerId = caffeLayerId + 1 ;
  end % For each layer
end % For each subnet

fclose(fileId) ;

%--------------------------------------------------------------------------
% Write the binary model file
%--------------------------------------------------------------------------
fprintf('Loading the prototxt...\n') ;
if p.useGpu
  caffe.set_mode_gpu() ;
  caffe.set_device(p.useGpu - 1) ;
else
  caffe.set_mode_cpu() ;
end

% Initialize a Caffe network from the prototxt we just created (with empty weights)
caffe.reset_all() ;
caffeNet = caffe.Net(prototxtFilename, 'test') ;

% Set its weights
fprintf('Writing weights...\n\t%s\n', modelFilename) ;
for n = 1:length(net) % For each subnet
  firstConv = true ;
  for l = 1:length(net{n})  
    if ~isfield(net{n}{l}, 'filters') || isempty(net{n}{l}.filters), continue ; end % No weights: skip layer

    % Rename for readability
    layerName = net{n}{l}.name;
    filters = net{n}{l}.filters ;
    biases = net{n}{l}.biases ;
    assert(ndims(filters) <= 4) ; % Can be < 4 if last dimension == 1

    filters = permute(filters, [2 1 3 4]) ; % Convert from convolution (MatConvNet's convention) to correlation (Caffe's convention) filters

    if firstConv % First convolutional layer at each scale
      if size(filters, 3) == 3
        filters = filters(:, :, [3 2 1], :) ; % Swap channel order to handle BGR images (Caffe's convention) instead of RGB ones (MatConvNet's convention) 
      end
      firstConv = false ; % Do this only once per scale
    end

    % Copy weights and biases into the Caffe model
    caffeNet.layers(layerName).params(1).set_data(filters) ;
    caffeNet.layers(layerName).params(2).set_data(biases(:)) ;
  end
end
caffeNet.save(modelFilename) ; % Save the Caffe model

%--------------------------------------------------------------------------
% Test the Caffe model
%--------------------------------------------------------------------------
if ~testCaffeModel
  return ; % Stop here
end

result_matlab = res{end}(net{end}{1}.outputLayer).x ; % Result produced by the Matlab implementation

% Run through the Caffe network
fprintf('Testing the Caffe model...\n') ;
caffeNet = caffe.Net(prototxtFilename, modelFilename, 'test') ;
inputDataCaffe = matlabImgToCaffe(inputData) ;
tmp = caffeNet.forward({inputDataCaffe}) ;
result_caffe = tmp{1};
result_caffe = permute(result_caffe, [2 1 3]) ; % Permute rows/columns from Caffe to Matlab format

% Display result
figure ; imshow(flowToColor(result_matlab)) ; title('Matlab result') ;
figure ; imshow(flowToColor(result_caffe)) ; title('Caffe result') ;

% Time the evaluation of the Caffe model
%%{
fprintf('Measring evaluation time of the Caffe model...\n') ;
tic
for i = 1:20
  tmp = caffeNet.forward({inputDataCaffe}) ;
end
time = toc ;
disp(time/20) % Time in seconds per evaluation
%}

%%{
keyboard
displayCnn(p, net, res) % Display Matlab network architecture
% Compare feature maps at the equivalent layers in the Matlab/Caffe models
ims(res{1}(2).x) ; ims(permute(caffeNet.blobs(['data']).get_data(), [2 1 3])) ; % Input
for s = 1:length(net) ; ims(res{s}(5).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer2']).get_data(), [2 1 3])) ; end % Rescaling
for s = 1:length(net) ; ims(res{s}(7).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer3']).get_data(), [2 1 3])) ; end % Motion Filters
for s = 1:length(net) ; ims(res{s}(8).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer4']).get_data(), [2 1 3])) ; end % Squaring
for s = 1:length(net) ; ims(res{s}(9).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer5']).get_data(), [2 1 3])) ; end % Pooling
for s = 1:length(net) ; ims(res{s}(10).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer6']).get_data(), [2 1 3])) ; end % Cross-channel normalization
for s = 1:length(net) ; ims(res{s}(12).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer7']).get_data(), [2 1 3])) ; end % Smoothing
for s = 1:length(net) ; ims(res{s}(13).x) ; ims(permute(caffeNet.blobs(['scale' num2str(s) '-layer8']).get_data(), [2 1 3])) ; end % Upsampling
ims(res{end}(14).x) ; ims(permute(caffeNet.blobs('decoding-layer1').get_data(), [2 1 3])) ; % Concat
ims(res{end}(15).x) ; ims(permute(caffeNet.blobs('decoding-layer2').get_data(), [2 1 3])) ; % Pixelwise decoding
ims(res{end}(17).x) ; ims(permute(caffeNet.blobs('decoding-layer3').get_data(), [2 1 3])) ; % Softmax
ims(res{end}(18).x) ; ims(permute(caffeNet.blobs('decoding-layer4').get_data(), [2 1 3])) ; % Projection as UV map
% Get filters
f = caffeNet.params('scale1-layer3', 1).get_data() ;
f = net{1}{6}.allFilters ;
f = caffeNet.params('decoding-layer2', 1).get_data() ;
f = caffeNet.params('decoding-layer2', 2).get_data() ; % Biases
%}

caffe.reset_all() ;

end

%==========================================================================
function writeStride(fileId, layer)
  if isfield(layer, 'stride')
    if length(layer.stride) == 1
      fprintf(fileId, '    stride: %d\n', layer.stride) ;
    elseif length(layer.stride) == 2
      fprintf(fileId, '    stride_h: %d\n', layer.stride(1)) ;
      fprintf(fileId, '    stride_w: %d\n', layer.stride(2)) ;
    end
  end
end
%==========================================================================
function writeKernelSize(fileId, kernelSize)
  if length(kernelSize) == 1
    fprintf(fileId, '    kernel_size: %d\n', kernelSize) ;
  elseif length(kernelSize) == 2
    fprintf(fileId, '    kernel_h: %d\n', kernelSize(1)) ;
    fprintf(fileId, '    kernel_w: %d\n', kernelSize(2)) ;
  end
end

%==========================================================================
function writePad(fileId, layer)
  if isfield(layer, 'pad')
    if length(layer.pad) == 1
      fprintf(fileId, '    pad: %d\n', layer.pad) ;
    elseif length(layer.pad) == 4
      fprintf(fileId, '    pad_h: %d\n', layer.pad(1)) ;
      fprintf(fileId, '    pad_w: %d\n', layer.pad(3)) ;
    else
      error('pad vector size must be 1 or 4')
    end
  end
end

%==========================================================================
function writeBlobNames(fileId, net, n, l)
  % The given n and l are the current net/layer
  % Find the source net/layer
  if isfield(net{n}{l}, 'src')
    bottomName = net{net{n}{l}.src(1)}{net{n}{l}.src(2)}.name ;
  else
    assert(l > 1) ;
    bottomName = net{n}{l-1}.name ;
  end
  if (n == length(net)) && (l == net{end}{1}.outputLayer) % Last net, output layer
    topName = 'output'; % Output blob
  else
    topName = net{n}{l}.name;
  end
  fprintf(fileId, '  top: "%s"\n', topName) ;
  fprintf(fileId, '  bottom: "%s"\n', bottomName) ;
end

%==========================================================================
function img = matlabImgToCaffe(img)
  img = single(img) ;
  img = permute(img, [2 1 3 4]) ; % Convert from HxWxCxN to WxHxCxN per Caffe's convention
  if size(img, 3) == 3
    img = img(:,:, [3 2 1], :) ; % Convert from RGB to BGR channel order per Caffe's convention
  end
end
