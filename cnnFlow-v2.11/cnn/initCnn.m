function net = initCnn(p, net)
%INITCNN Initialize CNN before running/training it.

if isempty(net), return ; end % Nothing to do

%--------------------------------------------------------------------------
% Network initialization
%--------------------------------------------------------------------------
for n = 1:numel(net)
  for l = 1:numel(net{n})
    if ~isfield(net{n}{l}, 'filters') && ~isfield(net{n}{l}, 'biases'), continue ; end % Skip non-trainable layers

    % Create fields, not all are really used, but this will allow running the same code in all cases
    if ~isfield(net{n}{l}, 'filters'),             net{n}{l}.filters = [] ; end
    if ~isfield(net{n}{l}, 'filtersMomentum'),     net{n}{l}.filtersMomentum = zeros(size(net{n}{l}.filters), 'single') ; end
    if ~isfield(net{n}{l}, 'filtersLearningRate'), net{n}{l}.filtersLearningRate = 0 ; end
    if ~isfield(net{n}{l}, 'filtersWeightDecay'),  net{n}{l}.filtersWeightDecay = 0 ; end
    if ~isfield(net{n}{l}, 'biases'),              net{n}{l}.biases = [] ; end
    if ~isfield(net{n}{l}, 'biasesMomentum'),      net{n}{l}.biasesMomentum = zeros(size(net{n}{l}.biases), 'single') ; end
    if ~isfield(net{n}{l}, 'biasesLearningRate'),  net{n}{l}.biasesLearningRate = 0 ; end
    if ~isfield(net{n}{l}, 'biasesWeightDecay'),   net{n}{l}.biasesWeightDecay = 0 ; end

    if isfield(net{n}{l}, 'weightProjection')
      P = net{n}{l}.weightProjection ;
      net{n}{l}.weightInverseProjection = sparse(P) ;
      net{n}{l}.weightInverseProjection = sparse(pinv(full(P))) ;
    end
  end
end

%--------------------------------------------------------------------------
% Determine which intermediate results can be erased and at which layer we can stop backpropagating early (for efficiency, when early layers are fixed/not trained)
%--------------------------------------------------------------------------
% Set initial default value
for n = 1:numel(net) % For each subnet
  net{n}{1}.earlyStopMaxLayer = +inf ;
end

for n = 1:numel(net) % For each subnet
  nLayers = numel(net{n}) ;

  for l = 1:nLayers % For each layer
    ly = net{n}{l} ;

    if isfield(ly, 'src') && (ly.src(2) == +inf)
      % Default source layer to the last one of that subnet
      net{n}{l}.src(2) = numel(net{ly.src(1)}) ;
      ly = net{n}{l} ; % Reload for using below
    end

    net{n}{l}.canErase = false ; % Set default value initially, maybe modified below

    % Find layer at which we can stop backpropagating
    if net{n}{1}.earlyStopMaxLayer == +inf
      % Check if we found a layer that needs to be reached during backprop
      % If we did: set layer ID at which to stop right after (in backward direction) this one (l-1)
      if any(strcmp(ly.type, {'conv', 'ori-conv', 'normalizel1'})) && ((ly.filtersLearningRate > 0) || (ly.biasesLearningRate > 0))
        net{n}{1}.earlyStopMaxLayer = l-1 ;
      elseif strcmp(ly.type, 'sharedconv') && ((net{ly.src(1)}{ly.src(2)}.filtersLearningRate > 0) || (net{ly.src(1)}{ly.src(2)}.biasesLearningRate > 0))
        net{n}{1}.earlyStopMaxLayer = l-1 ;
      elseif strcmp(ly.type, 'concat') || strcmp(ly.type, 'ori-concat')
        if l > 1
          net{n}{l-1}.canErase = true ; % Not sure about this one ?!
        end
        if net{ly.src(1)}{1}.earlyStopMaxLayer < +inf % If we have to backpropagate in the other "source" network of the concat/sum
          net{n}{1}.earlyStopMaxLayer = l-1 ; % Then we need to backpropagate this concat/sum layer
        end
      elseif isfield(ly, 'src')
        if l > 1
          net{n}{l-1}.canErase = true ; % Not sure about this one ?!
        end
        if net{ly.src(1)}{1}.earlyStopMaxLayer <= ly.src(2) % If we have to backpropagate in the other "source" network of the concat/sum
          net{n}{1}.earlyStopMaxLayer = l-1 ; % Then we need to backpropagate this copy layer
        end
      else
        if l > 1
          net{n}{l-1}.canErase = true ;
        end
      end
    end % if net{n}{1}.earlyStopMaxLayer == +inf

  end % For each layer
end % For each subnet

% Check for intermediate layers that cannot be erased because they are reused as auxiliary input to some special layers (with recurrent or "skip" connections)
for n = 1:numel(net) % For each subnet
  for l = 1:numel(net{n}) % For each layer
    ly = net{n}{l} ;
    switch ly.type
      case 'copy'
        net{ly.src(1)}{ly.src(2)}.canErase = false ;
      case 'flowloss'
        net{ly.src(1)}{ly.src(2)}.canErase = false ;
      case 'concat'
        net{ly.src(1)}{end}.canErase = false ;
      case 'ori-concat'
        net{ly.src(1)}{end}.canErase = false ;
      case 'warp'
        net{ly.uvMapSrc(1)}{ly.uvMapSrc(2)}.canErase = false ;
      case 'flowRecurrent'
        net{ly.backTo(1)}{ly.backTo(2)-1}.canErase = false ;
    end
  end % For each layer
end % For each subnet

% Never erase the network output and loss
net{end}{net{end}{1}.outputLayer}.canErase = false ; % Output
net{end}{end}.canErase = false ; % Loss
