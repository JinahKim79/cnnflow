function x = crop(x, h, w, position)
%CROP Crop the first two first dimensions of N-D array.
%   X = CROP(X, H, W) crops the two first dimensions of an array.

% Author: Damien Teney

if isempty(x)
  return; % Nothing to do
end

switch position
  case 'topleft'
    x = x(1:h, 1:w, :, :) ;
  case 'center'
    dH = size(x, 1) - h ; h1 = floor(dH / 2) ;
    dW = size(x, 2) - w ; w1 = floor(dW / 2) ;
    assert((dH >= 0) && (dW >= 0));
    x = x((1+h1) : (h1+h), (1+w1) : (w1+w), :, :) ;
  case 'left'
    dH = size(x, 1) - h ; h1 = floor(dH / 2) ;
    assert((dH >= 0));
    x = x((1+h1) : (h1+h), 1 : w, :, :) ;
  case 'right'
    dH = size(x, 1) - h ; h1 = floor(dH / 2) ;
    assert((dH >= 0));
    x = x((1+h1) : (h1+h), end-w+1 : end, :, :) ;
  case 'random'
    a = round(rand * (size(x, 1)-h)) ; b = round(rand * (size(x, 2)-w)) ; % Get a random position for the patch
    x = x((a+1):(a+h), (b+1):(b+w), :, :) ;
  otherwise, error('Invalid position parameter !') ;
end
