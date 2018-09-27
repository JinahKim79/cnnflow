function [out, n] = flow_measureError(p, x, c)
%FLOW_MEASUREERROR Compute various summed (over pixels and test images) error measures, between X and ground truth c (4D matrices). Also return the number of elements over which these error measure were summed (the mean error can then be recovered with the ratio of the two).

% No arguments: return the name of the error measures
if nargin <= 1
  out = {'EPE', 'AAE'} ;
  return ;
end

if numel(c) ~= numel(x)
  out = NaN ; n = 0 ;
  return ;
end

validMask = all(~isnan(c), 3) & all(~isnan(x), 3) ; % UV maps

%--------------------------------------------------------------------------
% End point error (average over pixels)
%--------------------------------------------------------------------------
% Average over pixels
err1 = sqrt(sum((x - c).^2, 3)) ; % Pixelwise endpoint error (L2 norm of difference vector)
out(1) = nansum(err1(:)) ;
n(1) = nnz(validMask) ;

%--------------------------------------------------------------------------
% Angular error (average over pixels)
%--------------------------------------------------------------------------
% Ref: http://www.scholarpedia.org/article/Optic_flow
getAngularError = @(uEst, vEst, uGt, vGt) (real(acos( (uGt(:).*uEst(:) + vGt(:).*vEst(:) + 1) ./ (sqrt(uGt(:).^2+vGt(:).^2+1) .* sqrt(uEst(:).^2+vEst(:).^2+1)) ))) ;
err2 = getAngularError(x(:, :, 1, :), x(:, :, 2, :), c(:, :, 1, :), c(:, :, 2, :)) ; % Pixelwise angular error
err2 = rad2deg(err2) ;
out(2) = nansum(err2(:)) ;
n(2) = nnz(validMask) ;
