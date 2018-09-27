function im = makeOrientedFiltersImage(f, nOrientationsIn, nOrientationsOut, makeMovie, addDelimiter)
%MAKEORIENTEDFILTERSIMAGE Wrapper for MAKEFILTERSIMAGE. Permute dimensions to have the orientations vary before the other feature channels.

% Author: Damien Teney

% Set default values for missing arguments
if nargin < 4
  makeMovie = 0 ;
end
if nargin < 5
  addDelimiter = 1 * (size(f, 1) > 1) ; % Add delimiter only with kernels of size > 1x1
end

if (ndims(f) ~= 4) && (ndims(f) ~= 3)
  displayWarning('Given filters do not have the right number of dimensions !\n') ;
  if nargout > 0, im = [] ; end
  return ;
end

% Automatically determine non-specified dimensions (empty arguments)
nChannelsIn  = size(f, 3) / nOrientationsIn ;
nChannelsOut = size(f, 4) / nOrientationsOut ;

f = reshape(f, size(f, 1), size(f, 2), nChannelsIn, nOrientationsIn, nChannelsOut, nOrientationsOut) ;
f = permute(f, [1 2 4 3 6 5]) ; % Permute dimensions to have the orientations vary before the other feature channels
f = reshape(f, size(f, 1), size(f, 2), size(f, 3)*size(f, 4), size(f, 5)*size(f, 6)) ;

if nargout == 0
  makeFiltersImage(f, false, true, makeMovie, addDelimiter) ; % Call without output argument
else
  im = makeFiltersImage(f, false, true, makeMovie, addDelimiter) ;
end
