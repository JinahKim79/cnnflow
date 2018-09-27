function result = getMirrorSymmetrizingMatrix(filterSize, nChannelsIn, nChannelsOut, enableLeftRight, enableUpDown, inputChannelsHaveOrientation)
%GETMIRRORSYMMETRIZINGMATRIX Make transformation matrix to create all mirror versions of vectorized (4D) filters.

assert(islogical(enableLeftRight)) ;
assert(islogical(enableUpDown)) ;

nOrientations = (1+enableLeftRight) * (1+enableUpDown) ;

if inputChannelsHaveOrientation
  result = zeros(filterSize, filterSize, nChannelsIn, nOrientations, nChannelsOut, nOrientations, filterSize, filterSize, nChannelsIn, nChannelsOut) ; % Orientation channels in input
else
  result = zeros(filterSize, filterSize, nChannelsIn,                nChannelsOut, nOrientations, filterSize, filterSize, nChannelsIn, nChannelsOut) ; % No orientation channels in input: expand into orientation at the same time as applying the filters
end

ori = 1 ;
for leftRight = 1:(1+enableLeftRight)
  for upDown = 1:(1+enableUpDown)
    % Fill up a transformation matrix that acts on 1 kernel
    tmp = zeros(filterSize, filterSize, filterSize, filterSize) ;
    for i = 1:filterSize, for j = 1:filterSize
      if leftRight == 1
        jj = j ;
      else
        jj = filterSize - j + 1 ; % Left/right flip
      end
      if upDown == 1
        ii = i ;
      else
        ii = filterSize - i + 1 ; % Up/down flip
      end
      tmp(ii, jj, i, j) = 1 ;
    end, end

    % Copy the same transformation matrix for all input/output channels
    for in = 1:nChannelsIn
      for out = 1:nChannelsOut
        if inputChannelsHaveOrientation
          result(:, :, in, ori, out, ori, :, :, in, out) = tmp ;
        else
          result(:, :, in,      out, ori, :, :, in, out) = tmp ;
        end
      end
    end

    ori = ori + 1 ;
  end
end

if inputChannelsHaveOrientation
  result = reshape(result, filterSize*filterSize*nChannelsIn*nOrientations*nChannelsOut*nOrientations, filterSize*filterSize*nChannelsIn*nChannelsOut) ;
else
  result = reshape(result, filterSize*filterSize*nChannelsIn*              nChannelsOut*nOrientations, filterSize*filterSize*nChannelsIn*nChannelsOut) ;
end

% Demo
%{
gradientKernel = repmat(0:8, 9, 1) ;
gradientKernel(1:3, :) = 0 ;
%gradientKernel = reshape(0:80, 9, 9) ;
f = zeros(9, 9, 2, 2) ;
f(:, :, 1, 1) = gradientKernel ;
f(:, :, 2, 2) = gradientKernel ;
makeFiltersImage(f)
f = reshape(f, 9*9*2*2, 1) ;
U = getMirrorSymmetrizingMatrix(9, 2, 2, true, true, false) ;
f2 = U * f ;
f2 = reshape(f2, 9, 9, 2, 2, []) ;
  f2(:, :, 1, 1, :)
  f2(:, :, 2, 2, :)
  f2(:, :, 1, 2, :)
f2 = reshape(f2, 9, 9, 2, []) ;
makeFiltersImage(f2)

% Inverse
f2 = reshape(f2, [], 1) ;
f1 = pinv(U) * f2 ;
f1 = reshape(f1, 9, 9, 2, 2) ;
makeFiltersImage(f1)
%}
