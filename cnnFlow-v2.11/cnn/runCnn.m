function [net, info] = runCnn(p, net, measureErrorFcn, processResultsFcn)
%RUNCNN Run the evaluation of a CNN on training/validation/test data. Can be used for evaluation only (then basically acting as a wrapper fo evalCnn), or for training with gradient descent (supports full-batch and stochastic mini-batch gradient descent).

% Author: Damien Teney

if ~exist(p.expDir, 'dir'), mkdir(p.expDir) ; end
modelPath = fullfile(p.expDir, 'net.mat') ;
modelPathTmp = fullfile(p.expDir, 'tmp.mat') ;
modelPathOld = fullfile(p.expDir, 'netOld.mat') ;
modelFigPath = fullfile(p.expDir, 'training.pdf') ;

assert(iscell(net)) ;

set(0, 'DefaultAxesFontSize', 6) ;

%--------------------------------------------------------------------------
% Initializations
%--------------------------------------------------------------------------
rng(0) ;

% Get the number of error measures
errorNames = measureErrorFcn(p) ;
nErrorMeasures = length(errorNames) ;

info = {} ;
res = {{}, {}, {}} ; % To store results over training/validation/test sets

assert((p.continue == 0) || (p.continue == 1) || (p.continue == 2)) ; % Only 3 possible values
if p.continue > 0
  % Continue with an existing result
  if ~exist(modelPath, 'file')
    error(sprintf('Cannot load/continue existing model: %s', modelPath)) ;
  end
  fprintf('Loading existing network: %s\n', modelPath) ;
  tmp = load(modelPath, 'info', 'net', 'p') ;
  % Compare the current parameters (p) and the loaded parameters (tmp.p)
  for f = fieldnames(p)'
    f = char(f) ;
    if isequal(f, 'runId') || isequal(f, 'runTrVaTe') || isequal(f, 'saveVisualizations') || isequal(f, 'sets') || isequal(f, 'epochLr') || isequal(f, 'nScenes') || isequal(f, 'files'), continue ; end % Do not display those fields even if they are different
    if isfield(tmp.p, f)
      a = getfield(tmp.p, f) ; b = getfield(p, f) ;
      if ~isequal(a, b)
        displayWarning('Different current/loaded parameter ''%s'':', f) ;
        if (numel(a) == 1) && (isinteger(a) || islogical(a))
          fprintf('  %d  %d\n', a, b) ;
        elseif (numel(a) == 1) && isfloat(a)
          fprintf('  %.3f  %.3f\n', a, b) ;
        elseif isvector(a) && isfloat(a(1))
          fprintf('\n') ;
          fprintf('  [') ; fprintf(' %.3f ', a(:)) ; fprintf(']\n') ;
          fprintf('  [') ; fprintf(' %.3f ', b(:)) ; fprintf(']\n') ;
        else
          fprintf('\n') ; disp(a) ; disp(b) ;
        end
      end
    end
  end
  if p.continue == 2
    fprintf('Using the loaded net as is\n') ;
    net = tmp.net ;
    net = initCnn(p, net) ; % Re-precompute some stuff just to be sure everything is correct/matching the current version of the code
  else
    net = copyCnnPretraining(p, net, tmp.net) ; % Copy trained layers
  end
  info = tmp.info ;
  displayWarning('Already trained for %d epochs\n', length(info{1}.objective)) ;
end

net = moveCnnGpu(p, net) ; % Make sure it's on the CPU/GPU as needed now (may have been trained differently)

% Empty initialization
for s = 1:3
  if isempty(info) || ~iscell(info) || (length(info) ~= 3) % Info not loaded or not loaded in a correct format (e.g. generated by an old version of the code)
    info{s}.speed = [] ;
    info{s}.objective = [] ;
    info{s}.lr = [] ;
    info{s}.n = zeros(0, nErrorMeasures) ;
    info{s}.error = zeros(0, nErrorMeasures) ;
  end
  if ~isfield(info{s}, 'lr'), info{s}.lr = [] ; end % In case we loaded a model that did not save the LR (from an old version of the code)
end

p.runTrVaTe = double(p.runTrVaTe) ; % Make sure it's numbers, not just logical values (0 if disabled, or the number between of iterations every which to run the tr/va/te)

% Initialize autostop (stops optimizaton if loss/error goes up)
if ~isempty(p.autoStop) && p.runTrVaTe(1)
  autoStopPreviousValue = +inf ; % Initialize the value to compare with
  assert(length(p.autoStop) == 3) ; % 3 values: min number of iterations, set to use (1-3: tr/va/te), error measure to use (0 for loss)
  assert(p.runTrVaTe(p.autoStop(2)) ~= 0) ; % Make sure we'll run the set to use for the autostop (tr/va/te)
end
stopNow = false ;

lastSaveTime = tic() ;
lastPlotTime = tic() ;

p.originalTrainingSet = p.sets{1} ; % Make a copy of the non-shuffled training set

%--------------------------------------------------------------------------
% Main code
%--------------------------------------------------------------------------
nEpochs = length(p.epochLr) ;
for e = 1:nEpochs % For each epoch
  fprintf('\nEpoch %d (LR = %.04f)\n', e, p.epochLr(e)) ;

  % Reset momentum if needed
  if (e == 1) ... % 1st iteration
  || ((e > 1) && (p.epochLr(e) ~= p.epochLr(e-1))) ... % LR just changed
  || p.fullBatchTraining % Full batch training ('momentum' is then used to accumulate gradient over mini batches)
    net = resetMomentum(net) ;
  end

  setNames = {'Training', 'Validation', 'Test'} ;
  for s = [1 2 3] % For training/validation/test sets
    isTraining = (s == 1) ;

    % Initialize (add an element for the current iteration)
    info{s}.speed(end+1) = 0 ;
    info{s}.objective(end+1) = 0 ;
    info{s}.lr(end+1) = 0 ;
    info{s}.error(end+1, 1:nErrorMeasures) = 0 ;
    info{s}.n(end+1, 1:nErrorMeasures) = 0 ; % Number of points over which the error has been computed (summed)

    if ~p.runTrVaTe(s) || (mod(e-1, p.runTrVaTe(s)) ~= 0) || ((s >= 2) && (p.runTrVaTe(s) > 1) && (e == 1)) % Run tr/va/te only every 'p.runTrVaTe(s)' iterations
      info{s}.objective(end) = NaN ;
      info{s}.lr(end) = NaN ;
      info{s}.error(end, 1:nErrorMeasures) = NaN ;
      continue ;
    end

    %----------------------------------------------------------------------
    % Loop over mini batches
    %----------------------------------------------------------------------
    nBatches = ceil(numel(p.sets{s})/p.batchSize) ;
    if isTraining && (nBatches > 1), p.sets{1} = p.originalTrainingSet(randperm(numel(p.originalTrainingSet))) ; end % Randomly shuffle training elements (useless if only 1 batch)
    for t = 1 : p.batchSize : numel(p.sets{s}) % For each batch (normally, p.startAtBatch=1)
      batchTime = tic ;

      batch = p.sets{s}( t : min(t+p.batchSize-1, numel(p.sets{s})) ) ; % Get indices of the input data for the current batch
      [res{s}, net] = evalCnn(p, net, batch, res{s}, isTraining, p.reuseIntermediateResults(s)) ; % Evaluate the CNN (forward and possibly backward)
      labels = gather(res{s}{1}(1).labels) ; % Make a copy of target labels (used for various things below)
      actualBatchSize = size(labels, 4) ;

      if p.displayNet && (e == 1) && (t == 1) % Very first iteration
        displayCnn(p, net, res{s}) ;
      end

      if isTraining && p.fullBatchTraining % Gradient step only at the end of each epoch (accumulate gradient over batches)
        %------------------------------------------------------------------
        % Accumulate gradient of the current batch
        %------------------------------------------------------------------
        for n = 1:numel(net) % For each subnet
          for l = 1 : numel(net{n}) % For each conv layer
            if ~isempty(res{n}(l).nShared) && (res{n}(l).nShared > 1)
              res{n}(l).dzdw{1} = res{n}(l).dzdw{1} ./ res{n}(l).nShared ; % Keep the average of a sum of gradients, accumulated due to sharing between layers or because of multiple (recurrent) evaluations
              res{n}(l).dzdw{2} = res{n}(l).dzdw{2} ./ res{n}(l).nShared ;
            end
            if net{n}{l}.filtersLearningRate > 0
              net{n}{l}.filtersMomentum = net{n}{l}.filtersMomentum - (p.epochLr(e) * net{n}{l}.filtersLearningRate) * res{n}(l).dzdw{1} ;
            end
            if net{n}{l}.biasesLearningRate > 0
              net{n}{l}.biasesMomentum = net{n}{l}.biasesMomentum - (p.epochLr(e) * net{n}{l}.biasesLearningRate) * res{n}(l).dzdw{2} ;
            end
          end % For each conv layer
        end % For each subnet

      elseif isTraining && ~p.fullBatchTraining % Gradient step at each batch
        %------------------------------------------------------------------
        % Minibatch gradient step
        %------------------------------------------------------------------
        for n = 1:numel(net) % For each subnet
          for l = 1 : numel(net{n}) % For each conv layer
            if ~isfield(net{n}{l}, 'filters') && ~isfield(net{n}{l}, 'biases'), continue ; end % No weights/biases to update

            if ~isempty(res{s}{n}(l).nShared) && (res{s}{n}(l).nShared > 1)
              % Keep the average of a sum of gradients, accumulated due to sharing between layers or because of multiple (recurrent) evaluations
              res{s}{n}(l).dzdw{1} = res{s}{n}(l).dzdw{1} ./ res{s}{n}(l).nShared ;
              res{s}{n}(l).dzdw{2} = res{s}{n}(l).dzdw{2} ./ res{s}{n}(l).nShared ;
            end

            if (net{n}{l}.filtersLearningRate > 0) && ~isempty(res{s}{n}(l).dzdw)
              net{n}{l}.filtersMomentum = ...
                p.momentum * net{n}{l}.filtersMomentum ...
                  - (p.epochLr(e) * net{n}{l}.filtersLearningRate) * (p.weightDecay * net{n}{l}.filtersWeightDecay) * net{n}{l}.filters ...
                  - (p.epochLr(e) * net{n}{l}.filtersLearningRate) / actualBatchSize * res{s}{n}(l).dzdw{1} ;
              net{n}{l}.filters = net{n}{l}.filters + net{n}{l}.filtersMomentum ;
              if isfield(net{n}{l}, 'allFilters'), net{n}{l}.allFiltersValid = false ; end % Mark the "full" set of filters (net{n}{l}.allFilters) as invalid, they do not match the reduced set anymore
            end

            if (net{n}{l}.biasesLearningRate > 0) && ~isempty(res{s}{n}(l).dzdw)
              net{n}{l}.biasesMomentum = ...
                p.momentum * net{n}{l}.biasesMomentum ...
                  - (p.epochLr(e) * net{n}{l}.biasesLearningRate) * (p.weightDecay * net{n}{l}.biasesWeightDecay) * net{n}{l}.biases ...
                  - (p.epochLr(e) * net{n}{l}.biasesLearningRate) / actualBatchSize * res{s}{n}(l).dzdw{2} ;
              net{n}{l}.biases = net{n}{l}.biases + net{n}{l}.biasesMomentum ;
              if isfield(net{n}{l}, 'allBiases'), net{n}{l}.allFiltersValid  = false ; end % Mark the "full" set of filters (net{n}{l}.allFilters) as invalid, they do not match the reduced set anymore
            end
          end % For each conv layer
        end % For each subnet
      end

      %--------------------------------------------------------------------
      % Display general information
      %--------------------------------------------------------------------
      batchTime = toc(batchTime) ;
      %wait(gpuDevice) ; % Might help avoid random CUDA crashes at the next line (?)
      outputLayerId = net{end}{1}.outputLayer ;
      predictions = gather(res{s}{end}(1+outputLayerId).x) ; % Get the prediction at the layer marked as the output (before the loss)
      [errorTmp, nTmp] = measureErrorFcn(p, predictions, labels) ;

      assert(isscalar(res{s}{end}(end).x)) ; % Make sure the loss is a scalar (average error over the batch elements)
      info{s}.objective(end)        = info{s}.objective(end) + double(gather(res{s}{end}(end).x)) ;
      info{s}.speed(end)            = info{s}.speed(end)     + batchTime ;
      info{s}.n(end, :)             = info{s}.n(end, :)      + nTmp ;
      info{s}.error(end, :)         = info{s}.error(end, :)  + errorTmp ;

      fprintf('%s, epoch %02d, batch %3d / %3d: ', setNames{s}, e, floor((t-1)/p.batchSize)+1, nBatches) ;
      fprintf(' %.1f images/s', actualBatchSize / batchTime) ;
      fprintf('   Error (%s)', strjoin(errorNames)) ; fprintf(' %.3f', errorTmp ./ nTmp) ; % Error over the last batch
      fprintf('   Loss %.3f', info{s}.objective(end) ./ actualBatchSize) ; % Loss over the last batch
      fprintf('\n') ;

      if p.saveVisualizations(s)
        %------------------------------------------------------------------
        % Make visualizations of network output
        %------------------------------------------------------------------
        %delete(sprintf('%s%s0*_*.png', p.expDir, filesep)) ;
        predictions = gather(res{s}{end}(1 + net{end}{1}.outputLayer).x) ; % Rename
        for i = 1:size(predictions, 4) % For each image in the batch (== length(batch) usually, but can be different if getBatch returned more augmentations than the number of original images)
          processResultsFcn(p, predictions(:, :, :, i), labels(:, :, :, i), false, batch(i)) ;
        end
      end % if p.saveVisualizations(s)

      if ~p.reuseIntermediateResults(s) % No reuse: can free up memory
        res{s} = {} ; % Not needed anymore
      end
    end % For each batch

    %----------------------------------------------------------------------
    % Full batch gradient step
    %----------------------------------------------------------------------
    if isTraining && p.fullBatchTraining % Training
      for n = 1:numel(net) % For each subnet
        for l = 1 : numel(net{n}) % For each conv layer
          if net{n}{l}.filtersLearningRate > 0
            net{n}{l}.filters = net{n}{l}.filters + net{n}{l}.filtersMomentum / numel(p.sets{s}) ; % Average gradients over all elements of all batches
            if isfield(net{n}{l}, 'allFilters'), net{n}{l}.allFilters = zeros(0, 'like', net{n}{l}.allFilters) ; end % Set as empty: do not match the (non-rotated) versions that were just updated
          end
          if net{n}{l}.biasesLearningRate > 0
            net{n}{l}.biases = net{n}{l}.biases + net{n}{l}.biasesMomentum / numel(p.sets{s}) ; % Average gradients over all elements of all batches
            if isfield(net{n}{l}, 'allBiases'), net{n}{l}.allBiases = zeros(0, 'like', net{n}{l}.allBiases) ; end % Set as empty: do not match the (non-rotated) versions that were just updated
          end
        end
      end
      net = resetMomentum(net) ; % Reset accumulated gradient for the next epoch
    end

    % Get mean values over all batches of the epoch
    info{s}.lr(end) = p.epochLr(e) ;
    info{s}.speed(end) = numel(p.sets{s}) / info{s}.speed(end) ;
    info{s}.objective(end) = info{s}.objective(end) / numel(p.sets{s}) ;
    info{s}.error(end, :) = info{s}.error(end, :) ./ info{s}.n(end, :) ;

    % Display average error over all batches
    if nBatches > 1
      fprintf('Average error (%s)', strjoin(errorNames)) ;
      fprintf(' %.3f', info{s}.error(end, :)) ;
      fprintf('   Loss %.3f', info{s}.objective(end)) ;
      fprintf('\n') ;
    end
  end % For training/validation/test sets

  if ~p.runTrVaTe(1) % Not training (= only 1 iteration): stop now
    break ;
  end

  % Erase "rotated" versions of the filters/biases to save disk space when saving the network
  %{
  for n = 1:numel(net) % For each subnet
    for l = 1 : numel(net{n}) % For each conv layer
      if isfield(net{n}{l}, 'allFilters'), net{n}{l}.allFilters = zeros(0, 'like', net{n}{l}.allFilters) ; end % Set empty
      if isfield(net{n}{l}, 'allBiases'),  net{n}{l}.allBiases  = zeros(0, 'like', net{n}{l}.allBiases ) ; end % Set empty
    end
  end
  %}

  %------------------------------------------------------------------------
  % Stop if loss/error goes up on validaton set
  %------------------------------------------------------------------------
  if ~isempty(p.autoStop) && p.runTrVaTe(1) && (mod(e-1, p.runTrVaTe(p.autoStop(2))) == 0) % Check the autostop criterion only if we just run the required tr/va/te set (e.g. if we only run the validation set every 'p.runTrVaTe(2)' epochs)
    if p.autoStop(3) == 0
      autoStopCurrentValue = info{p.autoStop(2)}.objective(end) ; % Use loss
    else
      autoStopCurrentValue = info{p.autoStop(2)}.error(end, p.autoStop(3)) ; % Use an error measure
    end
    if (e > p.autoStop(1)) && (autoStopCurrentValue > autoStopPreviousValue) % True if we've done enough iterations and the loss/error is increasing
      stopNow = true ;
      fprintf('Auto-stopping now !\n') ;
    end
    autoStopPreviousValue = autoStopCurrentValue ; % Save for comparison at the next iteration
  end

  %------------------------------------------------------------------------
  % Save
  %------------------------------------------------------------------------
  if (toc(lastSaveTime) > 60) || (e == nEpochs) || stopNow % Only every minute at most, or at the last epoch
    if p.runTrVaTe(1)
      fprintf('Saving net...\n') ;
      % Save
      if exist(modelPathTmp, 'file'), delete(modelPathTmp) ; end
      % Save in a temporary file (it can take some time, so if interrupted, only the temporary file is corrupted)
      save(modelPathTmp, 'net', 'info', 'p') ;
      if exist(modelPathOld, 'file'), delete(modelPathOld) ; end
      if exist(modelPath, 'file')
        java.io.File(modelPath).renameTo(java.io.File(modelPathOld)) ; % Rename the existing file instead of overwriting it
      end
      java.io.File(modelPathTmp).renameTo(java.io.File(modelPath)) ; % Rename the temporary file
      lastSaveTime = tic() ;
      fprintf('------------------------------------------------------------------------------------------------------------------------\n') ;
    end
  end

  %------------------------------------------------------------------------
  % Plot evolution of loss/error during training
  %------------------------------------------------------------------------
  if (nEpochs > 1) && ( (toc(lastPlotTime) > 10) || (e == nEpochs) ) % Only every 10 seconds at most, or at the last epoch
    nEpochsRun = length(info{1}.objective) ;
    es = max(1, (nEpochsRun-100+1)) : nEpochsRun ; % Epochs to plot
    figure(1) ; clf ;
    subplot(1, 1+nErrorMeasures, 1) ;
    plot(es, info{1}.objective(es), 'k.-') ; hold on ;
    plot(es, info{2}.objective(es), 'b.-') ;
    plot(es, info{3}.objective(es), 'r.-') ;
    xlabel('Epoch') ;
    grid on ;
    h = legend('train', 'val', 'test') ;
    set(h, 'color', 'none') ;
    title('Loss') ;
    xlim([min(es) max(es)+1]) ;

    for e = 1:nErrorMeasures % For each error measure
      subplot(1, 1+nErrorMeasures, 1+e) ;
      plot(es, info{1}.error(es, e), 'k.-') ; hold on ;
      plot(es, info{2}.error(es, e), 'b.-') ;
      plot(es, info{3}.error(es, e), 'r.-') ;
      h = legend('train','val', 'test') ;
      grid on ;
      xlabel('Epoch') ;
      set(h, 'color', 'none') ;
      title(['Error (', errorNames{e}, ')']) ;
      xlim([min(es) max(es)+1]) ;
    end

    set(1, 'PaperOrientation', 'landscape') ;
    set(1,'PaperPositionMode', 'manual', 'PaperUnits', 'centimeters', 'Paperposition', [1 1 28.7 20])
    drawnow ;
    try
      print(1, modelFigPath, '-dpdf') ;
    end% Also save as PDF
    lastPlotTime = tic() ;
  end

  if stopNow, break ; end
end  % For each epoch

%==========================================================================
function net = resetMomentum(net)
%RESETMOMENTUM Reset momentum of all layers

for n = 1:numel(net)
  for l = 1:numel(net{n})
    if isfield(net{n}{l}, 'filtersMomentum')
      net{n}{l}.filtersMomentum = zeros(1, 1, 'like', net{n}{l}.filtersMomentum) ; % Set to 0 and keep the same data type
    end
    if isfield(net{n}{l}, 'biasesMomentum')
      net{n}{l}.biasesMomentum = zeros(1, 1, 'like', net{n}{l}.biasesMomentum) ;
    end
  end
end
