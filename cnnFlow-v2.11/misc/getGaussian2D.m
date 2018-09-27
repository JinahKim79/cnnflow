function f = getGaussian2D(supportSize, sigma, normalize)
%GETGAUSSIAN2D Gaussian 2D filter.

% Author: Damien Teney

if nargin < 3 % Missing argument
  normalize = true ; % Set default value
end

s = floor((supportSize / 2)) ; 
[x, y] = meshgrid(-s:+s, -s:+s) ;

f = exp( -(x.*x + y.*y) / (2*sigma^2) ) ;
f(f < (eps * max(f(:)))) = 0 ;
if normalize
  % Make the filter sum to 1 exactly
  tmp = sum(f(:)) ;
  if tmp > 0
   f  = f / tmp ;
  end
else
  f = f / (2*pi*sigma*sigma) ;
end
