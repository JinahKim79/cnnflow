function [x, dzdw] = cnn_normalizel2(x, nGroups, groupDim, e, dzdy)
%CNN_NORMALIZEL2  CNN Element-wise unit L2 normalization.

% Author: Damien Teney

[h, w, c, n] = size(x) ;

groupSize = floor(c / nGroups) ;
if (groupSize * nGroups) ~= c
  error('The number of groups does not divide the number of channels !') ;
end

if isempty(dzdy)
  % Forward
  if groupDim == 4
    x = reshape(x, h, w, nGroups, groupSize, n) ; % Put the channels along which to normalize in its own dimension
    tmp = sqrt(sum(x.^2, 4)) + e ;
  elseif groupDim == 3
    x = reshape(x, h, w, groupSize, nGroups, n) ; % Put the channels along which to normalize in its own dimension
    tmp = sqrt(sum(x.^2, 3)) + e ;
  else
    error('Invalid argument ''groupDim'' !') ;
  end

  x = bsxfun(@rdivide, x, tmp) ;
  x = reshape(x, h, w, c, n) ; % Reshape result to the original dimensions

else
  % Backward
  x = x.^2 ;

  if groupDim == 4
    x = reshape(x, h, w, nGroups, groupSize, n) ;
    dzdy = reshape(dzdy, h, w, nGroups, groupSize, n) ;
    tmp = sum(x, 4) + e ;
  elseif groupDim == 3
    x = reshape(x, h, w, groupSize, nGroups, n) ;
    dzdy = reshape(dzdy, h, w, groupSize, nGroups, n) ;
    tmp = sum(x, 3) + e ;
  end

  x = bsxfun(@rdivide, bsxfun(@minus, tmp, x), tmp.^(3/2)) .* dzdy ;
end
