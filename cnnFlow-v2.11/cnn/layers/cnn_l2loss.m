function Y = cnn_l2loss(X,c,dzdy)
%CNN_L2LOSS  CNN L2 loss

% Author: Damien Teney

assert(isequal(size(c), size(X))) ;

if isempty(dzdy)
  % Forward pass
  tmp = (X - c).^2 ; % Pixelwise squared error
  Y = nanmean(tmp(:)) ; % Mean squared error
  Y = Y * size(X, 4) ; % We return a value summed over the batch (convention in runCnn())

else
  % Backprop
  %n = size(X, 1) * size(X, 2);
  %Y = 2 * ((X - c)) * dzdy  / n ;

  tmp = (X - c) ;
  n = sum(sum(all(~isnan(tmp), 3), 1), 2) ; % Number of pixels per image with a vector of non-NaN values (1 x 1 x 1 x nImages)
  Y = 2 * bsxfun(@times, tmp, 1 / n) * dzdy;

  Y(isnan(Y)) = 0 ;
end
