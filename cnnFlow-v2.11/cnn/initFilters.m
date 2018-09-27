function f = initFilters(h, w, c, d)
%INITFILTERS Random filter initialization with automatic scaling of values.
%  F = INITFILTERS(H, W, C, D) return a random filter of given dimensions
%  suitable for initialization, as proposed in:
%  Understanding the difficulty of training deep feedforward neural networks
%  Glorot Xavier and Bengio Yoshua, AISTAT 2010.

% Author: Damien Teney

f = randn(h, w, c, d, 'single') ; % Random numbers with normal distribution of sigma=1
scale = sqrt(1 / (h*w*c)) ; % Scale the values
f = scale .* f ;