function y = cnn_epeloss(x, c, dzdy)
%CNN_EPELOSS  CNN Optical flow EPE loss

% Author: Damien Teney

invalid2 = isnan(c) | isnan(x) ; % h x w x 2 x n
invalid = invalid2(:, :, 1, :) ; % h x w x 1 x n

if isempty(dzdy)
  % Forward pass
  tmp = sqrt(sum((x - c).^2, 3)) ; % Pixelwise EPE
  n = sum(sum(~invalid, 1), 2) ; % Number of pixels per image with defined ground truth and predictons (1 x 1 x 1 x nImages)
  tmp(invalid) = 0 ;
  tmp = sum(sum(tmp, 1), 2) ; % Average EPE per image (1 x 1 x 1 x nImages)
  y = sum(tmp ./ n) ; % Average per image, then sum over all images (returns a single scalar value)

else
  % Backprop
  tmp = (x - c) ;
  tmp2 = sum(tmp.*tmp, 3) ;
  %tmp2 = sqrt(tmp2) ; % Option 1
  tmp2 = sqrt(max(tmp2, 1e-6)) ; % Option 2: with normalization to avoid division by small values
  tmp = bsxfun(@rdivide, tmp, tmp2) ;
  n = sum(sum(~invalid, 1), 2) ; % Number of pixels per image with defined ground truth and predictions (1 x 1 x 1 x nImages)
  y = 2 .* bsxfun(@rdivide, tmp, n) .* dzdy ;

  % Don't propagate gradient where there were NaN values in the prediction or the ground truth
  y(invalid2) = 0 ;
end
