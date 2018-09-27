function Y = cnn_upsize(X, newSize, scaleFactor, scaleGradient, dzdy)
%CNN_UPSIZE  CNN Bilinear upsampling

% Author: Damien Teney

assert(scaleFactor >= 1) ;

if isempty(dzdy)
  % Forward
  [h, w, c, n] = size(X) ;
  if (scaleFactor == 1)
    %assert(isequal([h w], newSize)) ; Y = X ; % Nothing to do
    assert(max([h w] - newSize) <= 1) ; Y = fixSize(X, newSize(1), newSize(2)) ; % Allow mismatch by 1 pixel at most
    return ;
  end

  X = reshape(X, h, w, c*n) ; % Collapse the last 2 dimensions to process everything at once
  Y = imresize(X, scaleFactor, 'bicubic') ;
  Y = fixSize(Y, newSize(1), newSize(2)) ;
  Y = reshape(Y, size(Y, 1), size(Y, 2), c, n) ; % Restore the last 2 dimensions

else
  % Backward
  if (scaleFactor == 1) && isequal([size(X, 1) size(X, 2)], newSize), Y = dzdy ; return ; end % Nothing to do
  [h, w, c, n] = size(dzdy) ;

  dzdy = reshape(dzdy, h, w, c*n) ; % Collapse the last 2 dimensions to process everything at once
  Y = imresize(dzdy, 1/scaleFactor, 'bicubic') ;
  Y = fixSize(Y, size(X, 1), size(X, 2)) ;
  Y = reshape(Y, size(Y, 1), size(Y, 2), c, n) ; % Restore the last 2 dimensions

  switch scaleGradient
    case 2, Y = Y * (scaleFactor*scaleFactor) ; % Tends to equalize BIAS gradients across scales
    case 1, Y = Y * scaleFactor ; % Tends to equalize WEIGHT gradients across scales
    case 0 ; % Nothing to do
    otherwise, error('Invalid parameter ''scaleGradient'' !') ;
  end
end

%==========================================================================
function Y = fixSize(Y, newSize1, newSize2)
%FIXIZE Crop or pad feature map Y to a given height/width.

if size(Y, 1) > newSize1 % Too big: crop
  dH = size(Y, 1) - newSize1 ;
  h1 = floor(dH / 2) ; h2 = ceil(dH / 2) ;
  Y = Y(1+h1:end-h2, :, :, :) ;
end

if size(Y, 2) > newSize2 % Too big: crop
  dW = size(Y, 2) - newSize2 ;
  w1 = floor(dW / 2) ; w2 = ceil(dW / 2) ;
  Y = Y(:, 1+w1:end-w2, :, :) ;
end

if (size(Y, 1) < newSize1) || (size(Y, 2) < newSize2) % Too small: pad
  dH = newSize1 - size(Y, 1) ;
  dW = newSize2 - size(Y, 2) ;
  h1 = floor(dH / 2) ; h2 = ceil(dH / 2) ;
  w1 = floor(dW / 2) ; w2 = ceil(dW / 2) ;    
  Y = padarray(padarray(Y, [h1 w1], 'replicate', 'pre'), [h2 w2], 'replicate', 'post') ;
end
