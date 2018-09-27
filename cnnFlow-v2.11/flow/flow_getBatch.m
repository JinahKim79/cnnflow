function [images, targets, originalInputDimensions] = flow_getBatch(p, batch, enableAugmentation)
%FLOW_GETBATCH

if isempty(batch), return; end % Nothing to load

persistent images_loaded targets_loaded persistentRunId inputDimensions ;

if p.persistentBatchLoader && (isempty(images_loaded) || ~isequal(p.runId, persistentRunId))
  persistentRunId = p.runId ; % Make sure we reload the data each time we run the function main function (and getParams())
  fprintf('Preloading input data...\n') ;
  toLoad = 1 : length(p.files.im) ; % Load everything
elseif ~p.persistentBatchLoader
  toLoad = batch ; % Load the current batch
else
  toLoad = [] ;
end

if ~isempty(toLoad)
  % Get input file names
  files_im = strcat([p.dataDir filesep], cat(2, p.files.im{toLoad})', ['.png']) ;
  files_fl = strcat([p.dataDir filesep], cat(2, p.files.fl{toLoad})', ['.flo']) ;
  files_oc = strcat([p.dataDir filesep], cat(2, p.files.fl{toLoad})', ['_occlusions.png']) ;

  %------------------------------------------------------------------------
  % Load images
  %------------------------------------------------------------------------
  images_loaded = [] ;
  for i = 1 : length(files_im)
    if ~exist(files_im{i}, 'file'), continue ; end % Missing file: skip this image, it will stay as zero values
    tmp = imread(files_im{i}) ;

    if p.cropInput
      tmp = crop(tmp, p.cropInput(1), p.cropInput(2), 'topleft') ; % Crop to fixed size
    end

    if p.leftRightSplit == 1 % Crop to half width (left part for training, right part otherwise)
      if ismember(ceil(i / p.nFrames), p.sets{1}) % Training set
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'left') ;
      else % Validation/test set
        assert(ismember(ceil(i / p.nFrames), p.sets{2}) || ismember(ceil(i / p.nFrames), p.sets{3})) ;
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'right') ;
      end
    elseif p.leftRightSplit == 2 % Crop to half width (right part for training, left part otherwise)
      if ismember(ceil(i / p.nFrames), p.sets{1}) % Training set
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'right') ;
      else % Validation/test set
        assert(ismember(ceil(i / p.nFrames), p.sets{2}) || ismember(ceil(i / p.nFrames), p.sets{3})) ;
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'left') ;
      end
    end

    if isempty(images_loaded) % First iteration (not done outside the loop because we didn't know the dimensions of the images yet)
      images_loaded = zeros(size(tmp, 1), size(tmp, 2), size(tmp, 3), length(files_im), 'single') ;
    end
    if (size(tmp, 1) ~= size(images_loaded, 1)) || (size(tmp, 2) ~= size(images_loaded, 2)) || (size(tmp, 3) ~= size(images_loaded, 3))
      display(files_im{i}) ;
      error('Input images have different dimensions ! Try using batchSize=1 and non-persistent getBatch') ;
    end
    images_loaded(:, :, :, i) = single(tmp) ;
  end

  images_loaded = sum(images_loaded, 3) ./ (3*255) ; % Get grayscale images with values in [0,1]
  images_loaded = reshape(images_loaded, size(images_loaded, 1), size(images_loaded, 2), p.nFrames, []) ; % Concatenate series of frames along the 3rd dimension; for example  e.g. 3 frames R, then 3 frames G, then 3 frames B

  %------------------------------------------------------------------------
  % Load targets
  %------------------------------------------------------------------------
  targets_loaded = nan ; % One NaN element by default (in case there is no ground truth available)
  for i = 1 : length(files_fl)
    if ~exist(files_fl{i}, 'file'), continue ; end
    tmp = loadFloFile(files_fl{i}) ;

    if p.cropInput
      tmp = crop(tmp, p.cropInput(1), p.cropInput(2), 'topleft') ; % Crop to fixed size
    end

    if p.leftRightSplit == 1 % Crop to half width (left part for training, right part otherwise)
      if ismember(i, p.sets{1}) % Training set
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'left') ;
      else % Validation/test set
        assert(ismember(i, p.sets{2}) || ismember(i, p.sets{3})) ;
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'right') ;
      end
    elseif p.leftRightSplit == 2 % Crop to half width (right part for training, left part otherwise)
      if ismember(i, p.sets{1}) % Training set
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'right') ;
      else % Validation/test set
        assert(ismember(i, p.sets{2}) || ismember(i, p.sets{3})) ;
        tmp = crop(tmp, size(tmp, 1), size(tmp, 2)/2, 'left') ;
      end
    end

    targets_loaded(1:size(tmp, 1), 1:size(tmp, 2), 1:size(tmp, 3), i) = tmp ;
  end

  % Save the dimensions of the data before applying any downsampling
  inputDimensions = [size(images_loaded, 1), size(images_loaded, 2)];

  % Subsample images/targets if needed
  if p.stride(1) > 1, images_loaded = images_loaded(1:p.stride(1):end, 1:p.stride(1):end, :, :) ; end
  if p.stride(end) > 1, targets_loaded = targets_loaded(1:p.stride(end):end, 1:p.stride(end):end, :, :) ; end
end

if nargout == 0, return; end

if p.persistentBatchLoader
  % Return part of what was loaded
  images = images_loaded(:, :, :, batch) ;
  targets = targets_loaded(:, :, :, batch) ;
else
  % Return everything that was loaded
  images = images_loaded ;
  targets = targets_loaded ;
  clear images_loaded targets_loaded persistentRunId ;
end

if p.useGpu
  images = gpuArray(images) ;
  %targets = gpuArray(targets) ;
end

% Data augmentation (horizontal/vertical flips and 90deg rotations)
if enableAugmentation
%if true
  % Random flips
  %{
  nImages = size(images, 4) ;
  flips1 = rand(1, nImages) > .5 ; % Random logical values
  flips2 = rand(1, nImages) > .5 ;
  images(:, :, :, flips1) = flip(images(:, :, :, flips1), 1) ; % Flip up/down
  images(:, :, :, flips2) = flip(images(:, :, :, flips2), 2) ; % Flip left/right
  targets(:, :, :, flips1) = flip(targets(:, :, :, flips1), 1) ;
  targets(:, :, :, flips2) = flip(targets(:, :, :, flips2), 2) ;
  targets(:, :, 2, flips1) = -targets(:, :, 2, flips1) ; % Change the sign of the Y flow
  targets(:, :, 1, flips2) = -targets(:, :, 1, flips2) ; % Change the sign of the X flow
  %}
  % Random rotations
  %{
  assert(p.useGpu) ;
  targets = gpuArray(targets) ; % Necessary to use pagefun() in rotateTargets()
  rots = rand(1, nImages) > .5 ; % Random logical values
  images( :, :, :, rots) = rotateImages90(images( :, :, :, rots)) ;
  targets( :, :, :, rots) = rotateTargets(images( :, :, :, rots)) ;
  targets = gather(targets) ; % Had to be on the GPU for pagefun()
  %}
  % Add all flips
  %%{
  images = cat(4, images, flipImages(images, 0, 1), flipImages(images, 1, 0), flipImages(images, 1, 1)) ;
  targets = cat(4, targets, flipFlows(targets, 0, 1), flipFlows(targets, 1, 0), flipFlows(targets, 1, 1)) ;
  %}
  % Add all rotations
  %{
  images = cat(4, images, rotateImages90(images)) ;
  targets = cat(4, targets, rotateFlows90(targets)) ;
  %}
end

originalInputDimensions = inputDimensions ;

%=================================================================
% Utility function to generate augmentations

function images = flipImages(images, enableFlipLr, enableflipUd)
if enableFlipLr, images(:, :, :, :) = flip(images(:, :, :, :), 2) ; end % Flip left/right
if enableflipUd, images(:, :, :, :) = flip(images(:, :, :, :), 1) ; end % Flip up/down

function targets = flipFlows(targets, enableFlipLr, enableflipUd)
if enableFlipLr
  targets(:, :, :, :) = flip(targets(:, :, :, :), 2) ; % Flip left/right
  targets(:, :, 1, :) = -targets(:, :, 1, :) ; % Change the sign of the X flow
end
if enableflipUd
  targets(:, :, :, :) = flip(targets(:, :, :, :), 1) ; %  Flip up/down
  targets(:, :, 2, :) = -targets(:, :, 2, :) ; % Change the sign of the Y flow
end

function images = rotateImages90(images)
tmp = pagefun(@rot90, images) ;
images(:) = 0 ; % Fill in the "background" (empty regions in the rectangular canvas because of non-square images)
sz = min(size(images,  1), size(images,  2)) ;
images( 1:sz, 1:sz, :, rots) = tmp(1:sz, 1:sz, :, :) ; % Copy the square part that fits in the original canvas

function targets = rotateFlows90(targets)
tmp = pagefun(@rot90, targets) ;
targets(:) = nan ; % Fill in the "background" (empty regions in the rectangular canvas because of non-square images)
sz = min(size(targets, 1), size(targets, 2)) ;
targets(1:sz, 1:sz, :, rots) = tmp(1:sz, 1:sz, :, :) ;
targets(:, :, [1 2], rots) = targets(:, :, [2 1], rots) ; % Exchange U/V flow values
targets(:, :, 2, rots) = -targets(:, :, 2, rots) ; % Change the sign of the Y flow
