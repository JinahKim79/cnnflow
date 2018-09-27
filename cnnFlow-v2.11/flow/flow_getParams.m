function p = flow_getParams(userParams, p)
%FLOW_GETPARAMS Parse user input and return structure with given parameters, or default values when not specified.
% Also create output directories and parse input directories to build lists of input/ground truth files.

% Author: Damien Teney

if nargin < 2
  p = [] ; % Empty initialization
end

%--------------------------------------------------------------------------
% General parameters
%--------------------------------------------------------------------------
p.continue = 0 ; % 0: Create new network; 1: create new network then load trained layers (network definition should match the loaded one); 2: load existing network used as is
p.conserveMemory = true ;
p.sync = false ;
p.runTrVaTe = [1 1 0] ;
p.saveVisualizations = [0 1 1] ;
p.reuseIntermediateResults = [0 0 0] ; % Cache results of fixed (untrained) layers; 3 values for the tr/va/te dataset splits; only supported if there is only 1 batch, and no recurrent connections in the network
p.displayNet = true ;
p.useGpu = 1 ;

%--------------------------------------------------------------------------
% Training parameters
%--------------------------------------------------------------------------
p.fullBatchTraining = false ; % Mini/full batch training
p.nEpochs = 10000 ;
p.weightDecay = 0.0005 ;
p.momentum = 0.9 ;
p.autoStop = [] ; % If not used, leave empty; if used, must contain 3 values: min number of iterations, dataset split to use (1/2/3: tr/va/te), ID of error metric measure to use (0 for loss)

[p, userParams] = vl_argparse(p, userParams) ; % Load user input

%--------------------------------------------------------------------------
% Task-specific parameters
%--------------------------------------------------------------------------
p.modelId = 1 ;
p.displayWeights = false ;

% Architecture (enable/disable particular operations and layers)
p.bidirectionalFlow = false ; % Temporal extent of the motion filters
p.nFrames = 3 ; % Temporal extent of the motion filters
p.motionFilterWidth = 7 ; % Typically 9x9 (max for hardcoded filters), 7x7, or 5x5
p.enableStdNormalization = true ;
p.channelNormalization = 'L1' ; % L1, L2, or none
p.nRecurrentIterations = 1 ;
p.fixBordersSmoothing = true ; % Add mirror padding before applying smoothing filters
p.maskBorders = 1 ;
p.lossType = 0 ; % 0: Regression, 1: Classification

% Learning rates, weight decay, (re)initializations
p.lr(1:4) = 0 ;
p.lrBias(1:4) = 0 ;
p.wd(1:4) = 1 ;
p.randomInit(1:4) = 0 ;

p.nScales = 10 ;
p.nOris = 12 ;
p.nSpeeds = 2 ; 
p.nSpeedsDecoding = 12 ;
p.maxSpeed = 1.25 ;
p.maxSpeedDecoding = 35 ;

% Resolution of every feature map in the network, given as absolute strides/scale (relative to the original input, not relatve to the previous layer)
p.stride = [1 1 2 2 2] ;

[p, userParams] = vl_argparse(p, userParams) ; % Load user input

% Derive scales/orientations/speeds from the parameters defined above (or given by the user)
p.downscalings = ceil((1/.68).^(0:p.nScales-1)) ;
p.oris = [0:p.nOris-1] * (2*pi/p.nOris) ; % Orientation uniformly distributed on the circle
p.speeds = linspace(0, p.maxSpeed, p.nSpeeds) ; % Speed of the motion filters
p.speedsDecoding = 1e-6 + logspace(0, log10(p.maxSpeedDecoding+1), p.nSpeedsDecoding) - 1 ; % Output speeds (log-spaced)

p.visualizationQuiver = 0 ; % Overlay arrows onto the "colorwheel" visualizations of output flow maps
p.uniqueVisualizationMaxFlow = false ; % Normalize the max flow of all output visualizations by a same value (instead of using the max of each flow map)

%--------------------------------------------------------------------------
% Input/output parameters
%--------------------------------------------------------------------------
% Input/result directories (define absolute directories)
p.dataDir = [fileparts(mfilename('fullpath')), filesep, '..', filesep, 'data'] ; % ../data/
p.expDir = [fileparts(mfilename('fullpath')), filesep, '..', filesep, 'results', filesep, 'model', num2str(p.modelId)] ; % For example: ../results/model1/
if ~exist(p.expDir, 'dir'), mkdir(p.expDir) ; end

% Dataset parameters
p.persistentBatchLoader = true ; % Preload all images
p.batchSize = 8 ;
p.dataAugmentation = 0 ; % If == 1, enable augmentation at training time, if == 2, enable augmentation at training AND validation/test time
p.cropInput = [388, 584] ; % 
p.leftRightSplit = false ; % Use the left/right halves of the images for training/validation

[p, userParams] = vl_argparse(p, userParams) ; % Load user input

%--------------------------------------------------------------------------
% Get file names of input/ground truth data (will save file names that still need to have a prefix (input path) and suffix/extension added)
%--------------------------------------------------------------------------
p.scenes{1} = textread(fullfile(p.dataDir, 'trainingSet.txt'), '%s') ;
p.scenes{2} = textread(fullfile(p.dataDir, 'validationSet.txt'), '%s') ;
p.scenes{3} = textread(fullfile(p.dataDir, 'testSet.txt'), '%s') ;
flowsPerScene(1:3) = [1 1 1] ;

p.files.im = {} ;
p.files.fl = {} ;
p.sets = {[], [], []} ;
p.nScenes = [0 0 0] ;
for s = 1:3 % For training/validation/test sets
  if ~p.runTrVaTe(s)
    p.sets{s} = [] ;
    continue ;
  end

  for i = 1:numel(p.scenes{s}) % For each scene
    sceneName = p.scenes{s}{i} ; % Rename

    % Find images in the directory
    dirContents = dir(sprintf('%s%s%s%sframe_*.png', p.dataDir, filesep, sceneName, filesep)) ;
    frameIds = [] ;
    for fileId = 1:length(dirContents)
      tmp = sscanf(dirContents(fileId).name, 'frame_%s') ;
      if length(tmp) == 8
        frameIds(end + 1) = sscanf(dirContents(fileId).name, 'frame_%04u.png') ;
      end
    end
    frameIds = unique(frameIds) ;

    if length(dirContents) == 0, fprintf('\tCannot find images in: %s\n', sceneName) ; continue ; end
    if length(frameIds) < p.nFrames, fprintf('\tNot enough frames (%d, need %d) in: %s\n', length(frameIds), p.nFrames, sceneName) ; continue ; end

    % Find ground truth flows in the directory
    dirContents = dir(sprintf('%s%s%s%s*.flo', p.dataDir, filesep, sceneName, filesep)) ;
    frameIdsFlow = [] ;
    for fileId = 1:length(dirContents)
      frameIdsFlow(end + 1) = sscanf(dirContents(fileId).name, 'frame_%04u.flo') ;
    end
    frameIdsFlow = unique(frameIdsFlow) ;
    if isempty(frameIdsFlow), fprintf('\tSet %d: no ground truth flow available in: %s\n', s, sceneName) ; end

    if p.bidirectionalFlow
      f1 = floor(p.nFrames/2) ;
      f2 = ceil(p.nFrames/2) - 1 ;
    else
      f1 = 0 ;
      f2 = p.nFrames - 1 ;
    end

    % Normal case: use only frames for which we have enough images before/after
    firstUseableFrame = frameIdsFlow - f1 >= min(frameIds) ;
    lastUseableFrame  = frameIdsFlow + f2 <= max(frameIds) ;
    frameIdsUseable_fl = frameIdsFlow(firstUseableFrame & lastUseableFrame) ;

    if isempty(frameIdsUseable_fl) % No ground truth available: compute flow between for *all* frames (for which we have enough other frames before/after)
      frameIdsUseable_fl = frameIds(1+f2 : end-f2) ;
    end

    frameIdsUseable_fl = [10] ; % Special case for Middlebury: use only the 10th frame (comment this line if not desired)

    flowPerSceneCount = 0 ;
    for j = frameIdsUseable_fl(:)'
      frames_fl = j ;
      frames_im = (j - f1) : (j + f2) ;
      p.files.fl{end + 1} = cellfun(@(k) (sprintf('%s%sframe_%04d', sceneName, filesep, k)), num2cell(frames_fl), 'UniformOutput', false) ;
      p.files.im{end + 1} = cellfun(@(k) (sprintf('%s%sframe_%04d', sceneName, filesep, k)), num2cell(frames_im), 'UniformOutput', false) ;
      id = length(p.files.fl) ;
      p.sets{s}(end + 1) = id ;
      flowPerSceneCount = flowPerSceneCount + 1 ;
      if flowPerSceneCount >= (s), break ; end % Enough flows of the current scene are processed: stop here
    end

    p.nScenes(s) = p.nScenes(s) + 1 ;
  end % For each scene
end % For training/validation/test sets
fprintf('Using %d / %d / %d frames for training/validation/test.\n\n', length(p.sets{1}), length(p.sets{2}), length(p.sets{3})) ;

%--------------------------------------------------------------------------
% Derive hard-coded parameters from other ones set above
%--------------------------------------------------------------------------
p.epochLr = ones(1, p.nEpochs) ;
p.nScales = length(p.downscalings) ;
p.runId = clock() ; % Get a unique value

if p.bidirectionalFlow % Use frames before and after the frame of interest as input
  p.referenceFrame = ceil(p.nFrames / 2) ; % Middle frame
else % Use frames only after the frame of interest as input
  p.referenceFrame = 1 ; % First frame
end

if p.dataAugmentation == 1 && p.reuseIntermediateResults(1)
  displayWarning('Cannot use p.reuseIntermediateResults with data augmentation !\n') ;
  p.reuseIntermediateResults(1) = 0 ;
elseif p.dataAugmentation == 2 && any(p.reuseIntermediateResults(1:2))
  displayWarning('Cannot use p.reuseIntermediateResults with data augmentation !\n') ;
  p.reuseIntermediateResults(1:2) = 0 ;
end

for s = 1:3
  if p.reuseIntermediateResults(s) && ((numel(p.sets{s}) / p.batchSize) > 1)
    displayWarning(['Cannot use p.reuseIntermediateResults(' num2str(s) ') !\n']) ;
    p.reuseIntermediateResults(s) = 0 ; % p.reuseIntermediateResults is only supported if there is only 1 batch
    break ;
  end
end

if p.nRecurrentIterations > 1
  p.reuseIntermediateResults(1:3) = 0 ; % p.reuseIntermediateResults is only supported if there is no recurrent connections in the network
end

% Sanity checks
assert(length(p.stride) == 5) ;
assert(all(p.stride > 0)) ;
assert(issorted(p.stride(1:end-1))) ; % The stride must be increasing (feature maps getting coarser) at each layer
assert(all(rem(p.stride(2:4) ./ p.stride(1:3), 1) == 0)) ; % Make sure relative strides (from one layer to the next) are integer values
assert(p.stride(end) <= p.stride(end-1)) ; % The output can be the same as the previous feature map, or finer (upsampling is then applied at the end)
if p.stride(end) < p.stride(end-1), assert(p.lossType == 0) ; end % If we upsample at the end, we cannot use a classification loss (because of the way the architecture is wired)

% Check that we parsed all given arguments
if ~isempty(userParams)
  error(['Unknown argument(s): ' sprintf('%s ', userParams{1:2:end})]) ;
end
