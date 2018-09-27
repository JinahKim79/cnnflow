function varargout = getIntegerFactors(x)
%GETINTEGERFACTORS Integer factors.
%   [M N] = GETINTEGERFACTORS(X) returns numbers such that X = M * N.

% Author: Damien Teney

for m = floor(sqrt(x)) : -1 : 1
  if mod(x, m) == 0
    break;
  end
end
n = x / m ;

if nargout == 2
  varargout{1} = m;
  varargout{2} = n;
else
  varargout{1} = [m, n];
end
