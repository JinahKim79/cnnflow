function dst = copyCnnPretraining(p, dst, src)
%COPYCNNPRETRAINING Copy filter and biases of trainable layers from one model to another, leaving all other parameters (e.g. learning rates) untouched.

% Author: Damien Teney

if p.useGpu
  moveop = @(x) gpuArray(x) ;
else
  moveop = @(x) gather(x) ;
end

if numel(dst) ~= numel(src)
  warning('Different number of subnets in loaded/defined networks !') ;
end

for n = 1 : min(numel(dst), numel(src)) % For each subnet
  netSrc = src{n} ; netDst = dst{n} ; % Rename subnets
  assert(~iscell(netSrc{1}) && ~iscell(netDst{1})) ;

  if numel(netSrc) ~= numel(netDst)
    displayWarning('\tSubnet %u: Different number of layers ! (loaded: %d / model: %d)\n', n, numel(netSrc), numel(netDst)) ;
  end

  lSrc = 1 ; lDst = 1 ;
  while lDst < numel(netDst) % For each dst layer
    if lSrc > numel(netSrc)
      displayWarning('\tLayer %u.%d: not in the loaded net (only %d loaded layers / %d in model)\n', n, lDst, numel(netSrc), numel(netDst)) ;
      lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
    end

    if ~isequal(netSrc{lSrc}.type, netDst{lDst}.type)
      if numel(netSrc) == numel(netDst)
        displayWarning('\tLayer %u.%d (loaded: %s / model: %s): different types !\n', n, lDst, netSrc{lSrc}.type, netDst{lDst}.type) ;
        lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
      elseif numel(netSrc) < numel(netDst)
        displayWarning('\tModel layer %u.%u (%s): skipping, different type !\n', n, lDst, netDst{lDst}.type) ;
        lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
      elseif numel(netSrc) > numel(netDst)
        displayWarning('\tLoaded layer %u.%u (%s): skipping, different type !\n', n, lDst, netSrc{lSrc}.type) ;
        lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
      end
    end

    if ~isfield(netSrc{lSrc}, 'filters') && ~isfield(netSrc{lSrc}, 'biases') % Layer with no weights/biases to copy (e.g. a pooling layer)
      lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
    end

    if isfield(netDst{lDst}, 'doNotCopy') && netDst{lDst}.doNotCopy
      displayWarning('\tLayer %u.%d (%s): skipping, flagged "do not copy"\n', n, lDst, netSrc{lSrc}.type) ;
      lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
    end

    if ( isfield(netSrc{lSrc}, 'rotationInvariance')  && isfield(netDst{lDst}, 'rotationInvariance')  && netSrc{lSrc}.rotationInvariance  && ~netDst{lDst}.rotationInvariance ) % Was doing rotation invriance in the loaded model, but not in the current model (typically for fine-tuning without rotation invariance after pretraining with rotation inavriance)
      displayWarning('\tLayer %u.%d: rotationInvariance disabled now, copying ''allFilters''\n', n, l) ;
      netDst{lDst}.filters = moveop(netSrc{lSrc}.allFilters) ;
      netDst{lDst}.biases  = moveop(netSrc{lSrc}.allBiases) ;
      lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
    end

    if isfield(netSrc{lSrc}, 'filters') && isfield(netDst{lDst}, 'filters') && ~isequal(size(netSrc{lSrc}.filters), size(netDst{lDst}.filters))
      displayWarning('\tLayer %u.%d (%s): skipping, filters of different dimensions (loaded: %dx%dx%dx%d / model: %dx%dx%dx%d)\n', n, lDst, netSrc{lSrc}.type, size(netSrc{lSrc}.filters, 1), size(netSrc{lSrc}.filters, 2), size(netSrc{lSrc}.filters, 3), size(netSrc{lSrc}.filters, 4), size(netDst{lDst}.filters, 1), size(netDst{lDst}.filters, 2), size(netDst{lDst}.filters, 3), size(netDst{lDst}.filters, 4) ) ;
      lSrc = lSrc + 1 ; lDst = lDst + 1 ; continue ;
    end

    % Copy
    netDst{lDst}.filters = moveop(netSrc{lSrc}.filters) ;
    netDst{lDst}.biases  = moveop(netSrc{lSrc}.biases) ;
    if isfield(netSrc{lSrc}, 'allFilters'), netDst{lDst}.allFilters = moveop(netSrc{lSrc}.allFilters) ; end
    if isfield(netSrc{lSrc}, 'allBiases'), netDst{lDst}.allBiases = moveop(netSrc{lSrc}.allBiases) ; end

    lSrc = lSrc + 1 ; lDst = lDst + 1 ;
  end

  src{n} = netSrc ; dst{n} = netDst ; % Rename modified subnets
end
