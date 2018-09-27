function y = cnn_softmax(x, nGroups, groupDim, dzdy)
%CNN_SOFTMAX  CNN softmax
%    Y = CNN_SOFTMAX(X) applies the softmax operator the data X. X
%    has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    D can be thought of as the number of possible classes and the
%    function computes the softmax along the D dimension. Often W=H=1,
%    but this is not a requirement, as the operator is applied
%    convolutionally at all spatial locations.
%
%    DZDX = CNN_SOFTMAX(X, DZDY) computes the derivative DZDX of the
%    CNN otuoutwith respect to the input X given the derivative DZDY
%    with respect to the block output Y. DZDX has the same dimension
%    as X.

% Copyright (C) 2014 Andrea Vedaldi. Modified by Damien Teney
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

[h, w, c, n] = size(x) ;
groupSize = floor(c / nGroups) ;

if groupDim == 4
  x = reshape(x, h, w, nGroups, groupSize, n) ; % Put the channels along which to normalize in its own dimension
  e = exp(bsxfun(@minus, x, max(x, [], 4))) ;
  l = sum(e, 4) ;
elseif groupDim == 3
  x = reshape(x, h, w, groupSize, nGroups, n) ; % Put the channels along which to normalize in its own dimension
  e = exp(bsxfun(@minus, x, max(x, [], 3))) ;
  l = sum(e, 3) ;
else
  error('Invalid argument ''groupDim'' !') ;
end
y = bsxfun(@rdivide, e, l) ;

if isempty(dzdy)
  % Forward
  y = reshape(y, h, w, c, n) ; % Reshape result to the original dimensions

else
  % Backward
  if groupDim == 4
    dzdy = reshape(dzdy, h, w, nGroups, groupSize, n) ;
    y = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, 4)) ;
  elseif groupDim == 3
    dzdy = reshape(dzdy, h, w, groupSize, nGroups, n) ;
    y = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, 3)) ;
  else
    error('Invalid argument ''groupDim'' !') ;
  end

  y = reshape(y, h, w, c, n) ; % Reshape result to the original dimensions
end
