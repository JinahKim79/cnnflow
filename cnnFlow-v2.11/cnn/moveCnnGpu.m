function net = moveCnnGpu(p, net)
%MOVECNNGPU Move CNN to/from gpu.

if isempty(net), return ; end % Nothing to do

if p.useGpu
  moveop = @(x) gpuArray(x) ;
else
  moveop = @(x) gather(x) ;
end

for n = 1:numel(net) % For each subnet
  for l = 1:numel(net{n}) % For each layer
    if isfield(net{n}{l}, 'filters')
      net{n}{l}.filters = moveop(net{n}{l}.filters) ;
    end
    if isfield(net{n}{l}, 'biases')
      net{n}{l}.biases = moveop(net{n}{l}.biases) ;
    end
    if isfield(net{n}{l}, 'allFilters')
      net{n}{l}.allFilters = moveop(net{n}{l}.allFilters) ;
    end
    if isfield(net{n}{l}, 'allBiases')
      net{n}{l}.allBiases = moveop(net{n}{l}.allBiases) ;
    end
    if isfield(net{n}{l}, 'weightProjection')
      if issparse(net{n}{l}.weightProjection) && verLessThan('matlab', '8.5') % Matlab pre-R2015a
        net{n}{l}.weightProjection = gather(net{n}{l}.weightProjection) ; % Sparse arrays on GPU not supported pre-R2015a
      else
        net{n}{l}.weightProjection = moveop(net{n}{l}.weightProjection) ;
      end
    end
    if isfield(net{n}{l}, 'weightInverseProjection')
      if issparse(net{n}{l}.weightInverseProjection) && verLessThan('matlab', '8.5') % Matlab pre-R2015a
        net{n}{l}.weightInverseProjection = gather(net{n}{l}.weightInverseProjection) ; % Sparse arrays on GPU not supported pre-R2015a
      else
        net{n}{l}.weightInverseProjection = moveop(net{n}{l}.weightInverseProjection) ;
      end
    end
  end
end
