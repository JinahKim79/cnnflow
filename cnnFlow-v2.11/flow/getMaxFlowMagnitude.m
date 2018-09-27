function r = getMaxFlowMagnitude(uvMap)
%GETMAXFLOWMAGNITUDE  Max magnitude in a UV flow map.

% Author: Damien Teney

assert(ndims(uvMap) == 3);
assert(size(uvMap, 3) == 2);

u = uvMap(:, :, 1);
v = uvMap(:, :, 2);

r = sqrt(u.^2 + v.^2);
r = max(r(:));

if isnan(r) % Can happen if all values in input are NaNs
  r = 5 ; % Set default value
end
