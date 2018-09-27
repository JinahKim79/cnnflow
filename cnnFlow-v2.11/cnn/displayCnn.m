function net = displayCnn(p, net, res)
%DISPLAYCNN Display CNN architecture as a table, each layer as a column.

% Authors: Andrea Vedaldi, Damien Teney

if (nargin > 2) && ~isempty(res)
  if ~isstruct(res{1})
    for s = 1:3
      if ~isempty(res{s}), res = res{s} ; break ; end
    end
  end
  if ~isstruct(res{1})
    displayWarning('Invalid input: ''res'' must be a cell array of struct !\n') ;
    res = [] ;
    %return ;
  end
end

rowsToDisplay = {'layer', 'line', 'type', 'info', 'lrFilter', 'lrBias', 'stride', 'pad', 'line', 'mem'};
if (nargin > 2) && ~isempty(res)
  rowsToDisplay = {rowsToDisplay{:}, 'xmem', 'dxmem', 'line', 'xheight', 'xwidth', 'xd', 'dxheight', 'dxwidth', 'dxd', 'dxnShared'} ;
end

totalMemoryCpuParams = 0 ;
totalMemoryGpuParams = 0 ;
totalMemoryCpuData = 0 ;
totalMemoryGpuData = 0 ;

fprintf('\n') ;
for n = 1:numel(net) % For each subnetwork
  for row = rowsToDisplay % For each row to display
    switch char(row)
      case 'layer', s = sprintf('Subnet #%d', n) ;
      case 'line', s = '-----------' ;
      case 'type', s = '' ;
      case 'info', s = '' ;
      case 'stride', s = 'stride' ;
      case 'padding', s = 'pad' ;
      case 'inDim', s = 'in dim' ;
      case 'outDim', s = 'out dim' ;
      case 'mem', s = 'net KB C/G' ;
      case 'xheight', s = 'x height' ;
      case 'xwidth', s = 'x width' ;
      case 'xd', s = 'x d' ;
      case 'dxheight', s = 'dx height' ;
      case 'dxwidth', s = 'dx width' ;
      case 'dxd', s = 'dx d' ;
      case 'dxnShared', s = 'shared wt?' ;
      case 'xmem', s = 'x MB C/G' ;
      case 'dxmem', s = 'dx MB C/G' ;
      otherwise, s = char(row) ;
    end
    fprintf('|%11s', s) ;

    for l = 1:numel(net{n}) % For each network layer
      ly = net{n}{l} ;
      switch char(row)
        %------------------------------------------------------------------
        case 'line', s = '-------' ;
        %------------------------------------------------------------------
        case 'layer', s = sprintf('%d', l) ;
        %------------------------------------------------------------------
        case 'type'
          switch ly.type
            case 'getBatch', s = 'gtBatch';
            case 'sharedconv', s = 'shrdcnv';
            case 'normalizel1', s = 'norm-L1';
            case 'normalizel2', s = 'norm-L2';
            case 'pool', if strcmpi(ly.method, 'avg'), s = 'avgpool'; else s = 'maxpool'; end
            case 'clearborder', s = 'clrBord' ;
            case 'loss', s = 'logloss' ;
            case 'softmaxloss', s = 'smloss' ;
            case 'identity', s = 'identty' ;
            case 'flowRecurrent', s = 'flowRec';
            case 'ori-concat', s = 'oriCat';
            case 'ori-conv', s = 'oriConv';
            otherwise s = trimStr(ly.type, 7) ;
          end
        %------------------------------------------------------------------
        case 'info'
          % Update support
          support(1:2, l) = [1;1] ; % Default support
          switch ly.type
            case 'conv', support(1:2, l) = max([size(ly.filters, 1) ; size(ly.filters, 2)], 1) ;
            case 'ori-conv', support(1:2, l) = max([size(ly.filters, 1) ; size(ly.filters, 2)], 1) ;
            case 'pool', support(1:2, l) = ly.pool(:) ;
          end
          % Display misc info depending on layer type
          switch ly.type
            case 'conv', s = sprintf('%dx%d', support(1, l), support(2, l)) ;
            case 'ori-conv', s = sprintf('%dx%d', support(1, l), support(2, l)) ;
            case 'pool', s = sprintf('%dx%d', support(1, l), support(2, l)) ;
            case 'clearborder', s = [int2str(ly.width) 'px'] ; %num2str(ly.newValue)
            case 'sharedconv', s = ['#' int2str(ly.src(1)) '.' int2str(ly.src(2))] ;
            case 'copy', s = ['#' int2str(ly.src(1)) '.' int2str(ly.src(2))] ;
            case 'warp', s = ['#' int2str(ly.uvMapSrc(1)) '.' int2str(ly.uvMapSrc(2))] ;
            case 'scale', s = ['*' sprintf('%3.2f', ly.s)] ;
            case 'downsize', s = sprintf('%1.3f', ly.scaleFactor) ;
            case 'upsize', s = sprintf('%1.1f', ly.scaleFactor) ;
            case 'concat', s = ['#' int2str(ly.src(1)) '.' int2str(ly.src(2))] ;
            case 'ori-concat', s = ['#' int2str(ly.src(1)) '.' int2str(ly.src(2))] ;
            case 'flowRecurrent', s = [int2str(ly.nRecurrentIterations) 'iter'] ;
            otherwise, s = '';
          end
        %------------------------------------------------------------------
        case 'lrFilter'
          if isfield(ly, 'filtersLearningRate') && (ly.filtersLearningRate > 0)
            s = sprintf('%.4f', ly.filtersLearningRate) ;
          else
            s = '' ;
          end
        %------------------------------------------------------------------
        case 'lrBias'
          if isfield(ly, 'biasesLearningRate') && (ly.biasesLearningRate > 0)
            s = sprintf('%.4f', ly.biasesLearningRate) ;
          else
            s = '' ;
          end
        %------------------------------------------------------------------
        case 'stride'
          stride(1:2, l) = 1 ; % Default value, maybe modified below
          switch ly.type
            case {'conv', 'ori-conv', 'pool'}
              if numel(ly.stride) == 1
                stride(1:2, l) = ly.stride ;
              else
                stride(1:2, l) = ly.stride(:) ;
              end
              if all(stride(:, l) == stride(1, l))
                s = sprintf('%d', stride(1, l)) ;
              else
                s = sprintf('%dx%d', stride(1, l), stride(2, l)) ;
              end
            case 'smooth', s = ['guide ' int2str(ly.guideStride)] ;
            %case 'bilateral', s = ['g ' int2str(ly.guideStrideLowres) ' ' int2str(ly.guideStrideHighres)] ;
            otherwise, s = '' ;
          end
        %------------------------------------------------------------------
        case 'pad'
          switch ly.type
            case {'conv', 'ori-conv', 'pool'}
              if numel(ly.pad) == 1
                tmp(1:4) = ly.pad ;
              else
                tmp(1:4) = ly.pad(:) ;
              end
            otherwise, tmp(1:4) = 0 ;
          end
          if ~any(tmp(:) > 0)
            s = '' ;
          elseif all(tmp(:)==tmp(1))
            s = sprintf('pad %d', tmp(1)) ;
          else
            s = sprintf('%d,%dx%d,%d', tmp(1), tmp(2), tmp(3), tmp(4)) ;
          end
        %------------------------------------------------------------------
        case 'xheight'
          if length(res{n}) <= l || isempty(res{n}(l+1).x)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l+1).x, 1)) ;
          end
        %------------------------------------------------------------------
        case 'xwidth'
          if length(res{n}) <= l || isempty(res{n}(l+1).x)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l+1).x, 2)) ;
          end
        %------------------------------------------------------------------
        case 'xd'
          if length(res{n}) <= l || isempty(res{n}(l+1).x)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l+1).x, 3)) ;
          end
        %------------------------------------------------------------------
        case 'dxheight'
          if length(res{n}) <= l || ~isfield(res{n}(l+1), 'dzdx') || isempty(res{n}(l).dzdx)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l).dzdx, 1)) ;
          end
        %------------------------------------------------------------------
        case 'dxwidth'
          if length(res{n}) <= l || ~isfield(res{n}(l+1), 'dzdx') || isempty(res{n}(l).dzdx)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l).dzdx, 2)) ;
          end
        %------------------------------------------------------------------
        case 'dxd'
          if length(res{n}) <= l || ~isfield(res{n}(l), 'dzdx') || isempty(res{n}(l).dzdx)
            s = '' ;
          else
            s = sprintf('%3.0f', size(res{n}(l).dzdx, 3)) ;
          end
        %------------------------------------------------------------------
        case 'dxnShared'
          if length(res{n}) <= l || ~isfield(res{n}(l), 'nShared') || isempty(res{n}(l).nShared)
            s = '' ;
          else
            s = sprintf('%2.0ftimes', res{n}(l).nShared) ;
          end
        %------------------------------------------------------------------
        case 'mem'
          [a, b] = xmem(ly) ;
          mem(1:2, l) = [a ; b] ;
          if (a == 0) && (b == 0)
            s = '' ;
          else
            s = sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;
          end
        %------------------------------------------------------------------
        case 'xmem'
          if length(res{n}) <= l
            s = '' ;
          else
            [a,b] = xmem(res{n}(l+1).x) ;
            if (a == 0) && (b == 0)
              s = '' ;
            else
              s = sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;
            end
          end
        %------------------------------------------------------------------
        case 'dxmem'
          if length(res{n}) <= l
            s = '' ;
          else
            [a,b] = xmem(res{n}(l+1).dzdx) ;
            if (a == 0) && (b == 0)
              s = '' ;
            else
              s = sprintf('%.0f/%.0f', a/1024^2, b/1024^2) ;
            end
          end
        %------------------------------------------------------------------
      end
      fprintf('|%7s', s) ;
    end  % For each network layer
    fprintf('|\n') ;
  end % For each row to display

  [a, b] = xmem(net{n}) ;
  fprintf('Subnet memory:  %.1f/%1.f MB (parameters)', a/1024^2, b/1024^2) ;
  totalMemoryCpuParams = totalMemoryCpuParams + a ;
  totalMemoryGpuParams = totalMemoryGpuParams + b ;
  if (nargin > 2) && ~isempty(res)
    [a, b] = xmem(res{n}) ;
    fprintf('   %.1f/%1.f MB (data)', a/1024^2, b/1024^2) ;
    totalMemoryCpuData = totalMemoryCpuData + a ;
    totalMemoryGpuData = totalMemoryGpuData + b ;
  end
  fprintf('  (CPU/GPU)\n') ;

  fprintf('\n') ;
end % For each subnetwork

% Display memory use summed over all subnets
if numel(net) > 1 % If more than 1 subnet
  fprintf('Total memory:  %.1f/%1.f MB (parameters)', totalMemoryCpuParams/1024^2, totalMemoryGpuParams/1024^2) ;
  if (nargin > 2) && ~isempty(res)
    fprintf('   %.1f/%1.f MB (data)', totalMemoryCpuData/1024^2, totalMemoryGpuData/1024^2) ;
  end
  fprintf('  (CPU/GPU, MB)\n') ;
end

nWeights = 0 ;
nWeights2 = 0 ;
for n = 1:numel(net) % For each subnetwork
  for l = 1:numel(net{n}) % For each layer
    if isfield(net{n}{l}, 'substractFromSignal') && net{n}{l}.substractFromSignal, continue ; end % Usually a hardcoded layer
    if isequal(net{n}{l}.type, 'normalizel1'), continue ; end % Has a 'bias' field but usually a fixed parameter
    if isfield(net{n}{l}, 'filters'), nWeights = nWeights + numel(net{n}{l}.filters) ; end
    if isfield(net{n}{l}, 'biases'),  nWeights = nWeights + numel(net{n}{l}.biases) ; end
    if isfield(net{n}{l}, 'allFilters'), nWeights2 = nWeights2 + numel(net{n}{l}.allFilters) ; end
    if isfield(net{n}{l}, 'allBiases'),  nWeights2 = nWeights2 + numel(net{n}{l}.allBiases) ; end
  end
end
fprintf('Number of weights and biases: %u', nWeights) ;
if nWeights2 ~= nWeights
  fprintf(' (%u, %.1fx)\n', nWeights2, nWeights2/nWeights) ;
else
  fprintf('\n') ;
end

fprintf('\n') ;

%==========================================================================
function [cpuMem,gpuMem] = xmem(s, cpuMem, gpuMem)

if nargin <= 1 % Non-recursive call
  cpuMem = 0 ;
  gpuMem = 0 ;
end

if isstruct(s) 
  for f = fieldnames(s)'
    f = char(f) ;
    for i = 1:numel(s)
      [cpuMem, gpuMem] = xmem(s(i).(f), cpuMem, gpuMem) ;
    end
  end
elseif iscell(s)
  for i = 1:numel(s)
    [cpuMem, gpuMem] = xmem(s{i}, cpuMem, gpuMem) ;
  end
elseif isnumeric(s)
  if isa(s, 'single')
    mult = 4 ;
  else
    mult = 8 ;
  end
  if isa(s,'gpuArray')
    gpuMem = gpuMem + mult * numel(s) ;
  else
    cpuMem = cpuMem + mult * numel(s) ;
  end
end
