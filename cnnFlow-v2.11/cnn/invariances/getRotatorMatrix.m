function result = getRotatorMatrix(filterSize, nChannelsIn, nChannelsOut, nOrientations, useOnlyHalfCircle, inputChannelsHaveOrientation)
%GETROTATORMATRIX Make transformation matrix to rotate vectorized (4D) filters. Note that the resulting filters do not have any cross-talk between orientation groups.

if inputChannelsHaveOrientation
  result = zeros(filterSize, filterSize, nChannelsIn, nOrientations, nChannelsOut, nOrientations, filterSize, filterSize, nChannelsIn, nChannelsOut) ; % Orientation channels in input
else
  result = zeros(filterSize, filterSize, nChannelsIn,                nChannelsOut, nOrientations, filterSize, filterSize, nChannelsIn, nChannelsOut) ; % No orientation channels in input: expand into orientation at the same time as applying the filters
end

for ori = 1:nOrientations
  if useOnlyHalfCircle
    angle = (ori-1) * (pi) / nOrientations ; % Half circle
  else
    angle = (ori-1) * (2*pi) / nOrientations ; % Full circle
  end
  tmp = getSingleRotatorMatrix(filterSize, angle) ;
  tmp = reshape(tmp, filterSize, filterSize, filterSize, filterSize) ;
  for in = 1:nChannelsIn
    for out = 1:nChannelsOut
      if inputChannelsHaveOrientation
        result(:, :, in, ori, out, ori, :, :, in, out) = tmp ;
      else
        result(:, :, in,      out, ori, :, :, in, out) = tmp ;
      end
    end
  end
end

if inputChannelsHaveOrientation
  result = reshape(result, filterSize*filterSize*nChannelsIn*nOrientations*nChannelsOut*nOrientations, filterSize*filterSize*nChannelsIn*nChannelsOut) ;
else
  result = reshape(result, filterSize*filterSize*nChannelsIn*              nChannelsOut*nOrientations, filterSize*filterSize*nChannelsIn*nChannelsOut) ;
end

% Demo
%{
nOrientations = 4 ;

[fOdd, fEven] = makeMotionFilter(5, 0, -0.6, 1, 1, false, false, false) ;
f = cat(4, fOdd, fEven) ;
makeFiltersImage(f)
f = reshape(f, 9*9*5*2, 1) ;
U = getRotatorMatrix(9, 5, 2, nOrientations, false, false) ;
f2 = U * f ;
f2 = reshape(f2, 9, 9, 5, 2, nOrientations) ;
f2 = reshape(f2, 9, 9, 5, 2*nOrientations) ;
makeFiltersImage(f2)

% With input channels that already have an orientation
U = getRotatorMatrix(9, 5, 2, nOrientations, false, true) ;
f2 = U * f ;
f2 = reshape(f2, 9, 9, 5, nOrientations, 2, nOrientations) ;
f2 = reshape(f2, 9, 9, 5*nOrientations, 2*nOrientations) ;
makeFiltersImage(f2)

% Inverse
f2 = reshape(f2, [], 1) ;
f1 = pinv(U) * f2 ;
f1 = reshape(f1, 9, 9, 5, 2) ;
makeFiltersImage(f1)
%}

%==========================================================================
function result = getSingleRotatorMatrix(sz, angle)
%GETSINGLEROTATORMATRIX Transformation matrix to rotate a vectorized image, for 1 specific angle.

center = 1 + ([sz, sz] - 1) / 2;
angle = -angle ; % Convention
R = [ +cos(angle) +sin(angle) ; -sin(angle) +cos(angle) ]';
result = zeros(sz, sz, sz+2, sz+2);
for jj = 1:sz
  for ii = 1:sz
    M = center + ([ii, jj] - center) * R; % Source indices
    if any(M < 0) || any(M > [sz sz]+1)
      continue;
    else
      % Compute the relative areas with the 4 surrounding pixels
      % Triangle area mapping
      % Reference: http://www.leptonica.com/rotation.html
      %{
      C = ceil(M);
      F = floor(M);
      result(ii, jj, F(1), F(2)) = (C(2)-M(2)) * (C(1)-M(1)) ;
      result(ii, jj, F(1), C(2)) = (M(2)-F(2)) * (M(1)-F(1)) ;
      result(ii, jj, C(1), F(2)) = (C(2)-M(2)) * (M(1)-F(1)) ;
      result(ii, jj, C(1), C(2)) = (M(2)-F(2)) * (C(1)-M(1)) ;
      %}
      % Bilinear interpolation
      %%{
      F = floor(M);
      s = (M-F); t = (1-s);
      F = F + 1;
      C = F + 1;
      result(ii, jj, F(1), F(2)) = t(1) * t(2) ;
      result(ii, jj, F(1), C(2)) = t(1) * s(2) ;
      result(ii, jj, C(1), F(2)) = s(1) * t(2) ;
      result(ii, jj, C(1), C(2)) = s(1) * s(2) ;
      %}
    end
  end
end
result = reshape(result(:, :, 2:end-1, 2:end-1), sz*sz, sz*sz);
