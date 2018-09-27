function res = moveResGpu(p, res)
%MOVERESGPU Move results of a CNN to/from gpu.

if isempty(res), return ; end % Nothing to do

if p.useGpu
  moveop = @(x) gpuArray(x) ;
else
  moveop = @(x) gather(x) ;
end

for n = 1:numel(res) % For each subres
  for l = 1:numel(res{n}) % For each layer
    if isfield(res{n}(l), 'x')
      res{n}(l).x = moveop(res{n}(l).x) ;
    end
    if isfield(res{n}(l), 'dzdx')
      %res{n}(l).dzdx = moveop(res{n}(l).dzdx) ;
    end
    if isfield(res{n}(l), 'dzdw')
      %res{n}(l).dzdw{1} = moveop(res{n}(l).dzdw{1}) ;
      %res{n}(l).dzdw{2} = moveop(res{n}(l).dzdw{2}) ;
    end
  end
end
