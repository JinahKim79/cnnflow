function mask = getDiskMask(diameter)
%GETDISKMASK Make circular binary mask of given odd diameter.
%   MASK = GETDISKMASK(SZ) returns a SZ x SZ matrix of true values inside a
%   circle or odd diameter SZ, false outside.

% Author: Damien Teney

assert(mod(diameter, 2) == 1) ; % Check we are given an odd diameter
radius = floor(diameter / 2) ; % Get radius from diameter
[x, y] = meshgrid(-radius:radius, -radius:radius) ;
d = sqrt(x.^2+y.^2) ; % Make 2D matrix of distances from the center of the matrix
mask = (d <= radius) ;
