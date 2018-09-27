function y = cnn_aaeloss(x, c, dzdy)
%CNN_AAELOSS  CNN Optical flow angular error loss

% Author: Damien Teney

% For debug
assert(size(x, 3) == 2) ;

invalid2 = isnan(c) | isnan(x) ; % h x w x 2 x n
%x(invalid2) = 0 ; c(invalid2) = 0 ; % Replace NaNs by 0s (necessary ??)
invalid = invalid2(:, :, 1, :) ; % h x w x 1 x n

if isempty(dzdy)
  % Forward pass

  % Computer the angular error
  % i.e. acos( (s*u + t*v + 1) / (sqrt(s^2+t^2+1) * sqrt(u^2+v^2+1)) )
  u = x(:, :, 1, :) ; v = x(:, :, 2, :) ; % Rename for clarity
  s = c(:, :, 1, :) ; t = c(:, :, 2, :) ;  
  tmp1 = (s.*u + t.*v + 1) ./ (sqrt(s.^2+t.^2+1) .* sqrt(u.^2+v.^2+1)) ;
  tmp1 = max(min(tmp1, +1), -1) ; % Remove numerical errors for acos()
  %tmp1 = gather(tmp1) ;
  tmp = acos(tmp1) ; % Pixelwise angular error
  tmp = rad2deg(tmp) ;

  % Average over (defined) pixels
  tmp(invalid) = 0 ;
  n = sum(sum(~invalid, 1), 2) ; % Number of pixels per image with defined ground truth and predictons (1 x 1 x 1 x nImages)
  tmp = sum(sum(tmp, 1), 2) ./ n ; % Average error per image (1 x 1 x 1 x nImages)
  y = sum(tmp) ; % Sum over all images (scalar value)

else
  % Backprop
  u = x(:, :, 1, :) ; v = x(:, :, 2, :) ;
  s = c(:, :, 1, :) ; t = c(:, :, 2, :) ;
  du = (-s .* v.^2 - s + t .* u .* v + u) ./ (sqrt(s.^2+t.^2+1) .* (u.^2+v.^2+1).^(3/2) .* sqrt(max(0, 1-(s .* u + t .* v + 1).^2 ./ ((s.^2+t.^2+1) .* (u.^2+v.^2+1))))) ;
  dv = -(t ./ (sqrt(s.^2+t.^2+1) .* sqrt(u.^2+v.^2+1)) - (v .* (s .* u + t .* v + 1)) ./ (sqrt(s.^2 + t.^2 + 1) .* (u.^2 + v.^2 + 1).^(3/2))) ./ sqrt(max(0, 1-(s .* u + t .* v + 1).^2 ./ ((s.^2+t.^2+1) .* (u.^2+v.^2+1)))) ;
  y = cat(3, du, dv) ;

  n = sum(sum(~invalid, 1), 2) ; % Number of pixels per image with defined ground truth and predictions (1 x 1 x 1 x nImages)
  y = bsxfun(@rdivide, y, n) ;
  y(isinf(y)) = 0 ;

  y = y .* dzdy ;

  % Don't propagate gradient where there were NaN values in the prediction or the ground truth
  y(invalid2) = 0 ;

  %if any(isnan(y(:))), keyboard ; end % For debug
  %if any(isinf(y(:))), keyboard ; end % For debug
end
