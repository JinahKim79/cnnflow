function [x, dzdw] = cnn_normalizel1(x, s, nGroups, e, groupDim, dzdy)
%CNN_NORMALIZEL1  CNN Element-wise unit L1 normalization (scale so each group of channel of each pixel sum to the given value 's'; 'e' is a noise floor to avoid numerical instabilities). Expand groups of channels along the 3rd or 4th dimension (see code).

% Author: Damien Teney

assert(~isempty(x)) ;
[h, w, c, n] = size(x) ;

groupSize = floor(c / nGroups) ;
if (groupSize * nGroups) ~= c
  error('The number of groups does not divide the number of channels !') ;
end

if isempty(dzdy)
  % Forward
  if groupDim == 4
    x = reshape(x, h, w, nGroups, groupSize, n) ; % Put the channels along which to normalize in its own dimension
    tmp = sum(x, 4) ;
  elseif groupDim == 3
    x = reshape(x, h, w, groupSize, nGroups, n) ; % Put the channels along which to normalize in its own dimension
    tmp = sum(x, 3) ;
  else
    error('Invalid argument ''groupDim'' !') ;
  end

  x = bsxfun(@rdivide, x, (tmp + e*groupSize)) ;
  x = reshape(x, h, w, c, n) ; % Reshape result to the original dimensions
  x = x .* s ;

else
  % Backward
  if groupDim == 4
    x = reshape(x, h, w, nGroups, groupSize, n) ;
    dzdy = reshape(dzdy, h, w, nGroups, groupSize, n) ;
    tmp = sum(x, 4) + e*groupSize ;
  elseif groupDim == 3
    x = reshape(x, h, w, groupSize, nGroups, n) ;
    dzdy = reshape(dzdy, h, w, groupSize, nGroups, n) ;
    tmp = sum(x, 3) + e*groupSize ;
  end

  % Derivative wrt 'e'
  if nargout > 1
    dzdw = -bsxfun(@rdivide, x, tmp.^2) .* dzdy .* s .* groupSize ;
    dzdw = mean(dzdw(:)) * n ; % The derivative is by convention (in runCnn) summed over the images, hence *n
    %dzdw = sum(dzdw(:)) ;
  end

  % Derivative wrt input 'x'
  x = bsxfun(@rdivide, bsxfun(@minus, tmp, x), tmp.^2) .* dzdy ;
  x = reshape(x, h, w, c, n) ; % Reshape result to the original dimensions
  x = x .* s ;
end
