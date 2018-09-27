function result = getIsotropizingMatrix(filterSize, nChannelsIn, nChannelsOut, nOrientations)
%GETISOTROPIZINGMATRIX Make transformation matrix to create circularly symmetric (isotropic) filters from their half cross-sections.

% Author: Damien Teney

result = zeros(filterSize, filterSize, nChannelsIn, nOrientations, nChannelsOut, nOrientations, ...
  ceil(filterSize/2), nChannelsIn, nChannelsOut) ; % Orientation channels in input

tmp = getSingleIsotropizingMatrix(filterSize) ;
tmp = reshape(tmp, filterSize, filterSize, []) ;
for ori = 1:nOrientations
  for in = 1:nChannelsIn
    for out = 1:nChannelsOut
      result(:, :, in, ori, out, ori, :, in, out) = tmp ;
    end
  end
end

result = reshape(result, filterSize*filterSize*nChannelsIn*nOrientations*nChannelsOut*nOrientations, ...
  ceil(filterSize/2)*nChannelsIn*nChannelsOut) ;

% Demo of the use of the result of this function
%{
filterSize = 9 ;
nOrientations = 3 ;
f = zeros(ceil(9/2), 2, 2) ;
f(:, 1, 1) = [0 1 0 2 3] ;
f(:, 2, 2) = [5 4 3 2 1] ;
makeFiltersImage(f)
f2 = getIsotropizingMatrix(filterSize, 2, 2, nOrientations) * f(:) ;
f2 = reshape(f2, filterSize, filterSize, 2*nOrientations, 2*nOrientations) ;
makeOrientedFiltersImage(f2, nOrientations, nOrientations) ;
%}

%==========================================================================
function result = getSingleIsotropizingMatrix(filterSize)

assert(isOdd(filterSize)) ;
r = ceil(filterSize / 2) ; % Radius

[x, y] = meshgrid(-(r-1):(r-1), -(r-1):(r-1));
distFromCenter = sqrt(x.^2 + y.^2) ; % 2D Matrix of indices wrt the center of the matrix

result = zeros(filterSize, filterSize, r) ;
for i = 1:r
  v = zeros(r, 1) ; v(i) = 1 ; % Make a 1-hot vector of size 'r'
  result(:, :, i) = interp1(0:r-1, v, distFromCenter, 'lin', 0) ;
end
result = reshape(result, filterSize*filterSize, r) ;
