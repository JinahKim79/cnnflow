function [out, aux1, aux2] = cnn_flowloss(x, labels_uv, vals_uv, dzdy, aux1, aux2)
%CNN_FLOWLOSS  CNN Flow classification loss
%   Usage forward:   [loss, aux1, aux2] = cnn_flowloss2(x, labels_uv, vals_uv, [], [])
%   Usage backward:  dzdx = cnn_flowloss2(x, labels_uv, vals_uv, dzdy, aux1, aux2)
%   x: values produced by the network (UV flow vectors)
%   labels_uv: ground truth values
%   vals_uv: discrete set of possible output values (the network performs classification into one of those values of each pixel)
%   aux1, aux2: intermediate values cached from the forward pass to the backward pass for speed

% Author: Damien Teney

sz = size(x) ;
if numel(sz) < 4, sz(4) = 1 ; end % Happens if only 1 image in x
nans = any(isnan(x), 3) | any(isnan(labels_uv), 3) ;

labels_uv(isinf(labels_uv)) = 999 ;

if isempty(dzdy)
  % Forward
  if isa(x, 'gpuArray'), vals_uv = gpuArray(vals_uv) ; end
  vals_uv = shiftdim(vals_uv, -2) ; % Dimensions: 1 x 1 x noutputValues x 2

  % Find values closest to the labels
  tmp = bsxfun(@minus, reshape(labels_uv, sz(1), sz(2), 1, 2, sz(4)), vals_uv) ;
  tmp = sum(tmp.^2, 4) ; % Squared Euclidean distance
  labels_indicator = bsxfun(@eq, tmp, min(tmp, [], 3)) ; % Binary indicator matrix of the best matching row in outputValues (h x w x noutputValues x 1 x n)
  labels_indicator = squeeze(labels_indicator) ;
  labels_indicator = bsxfun(@rdivide, labels_indicator, sum(labels_indicator, 3)) ; % Make sure the values sum to 1 (necessary for cases where the value just as close to several possible UV values)
  labels_indicator(repmat(nans, 1, 1, 2, 1)) = 0 ;

  % Softmax on input
  tmp = exp(bsxfun(@minus, x, max(x, [], 3))) ;
  x_soft = bsxfun(@rdivide, tmp, sum(tmp, 3)) ; 

  aux1 = x_soft ; aux2 = labels_indicator ; % Save intermediate results for the backward pass

  % Alternative for debug: loss = 1 - (ratio of pixels correctly classified, i.e. in the closest bin)
  %%{
  x_hard = ones(1, 1, 'like', x) * bsxfun(@eq, x, max(x, [], 3)) ; % Hardmax
  nans = repmat(nans, 1, 1, size(x, 3), 1) ; % Now has the same dimensions as x_hard and labels_indicator
  tmp = (x_hard == labels_indicator) ;
  out = sz(3) * (1 - nnzRatio(tmp(~nans))) ;
  return ;
  %}

  % Compute the loss
  loss = log(max(1e-16, 1-abs(x_soft - labels_indicator))) ; % Robust version (never returns -inf) of: loss = log(1-abs(x_soft - labels_indicator)) ;
  loss(repmat(nans, 1, 1, size(loss, 3), 1)) = 0 ; % Undefined labels
  assert(gather(all(loss(:) <= 0))) ; % Debug check
  assert(gather(all(~isnan(loss(:))))) ; % Debug check
  assert(gather(all(~isinf(loss(:))))) ; % Debug check
  loss = -sum(loss(:)) ;
  loss = loss / (sz(1)*sz(2)) ; % Average loss per pixel
  out = loss ; % Return the loss

else
  % Backward
  x_soft = aux1  ; labels_indicator = aux2 ; % Rename stuff saved from the forward pass

  % Compute dzdx
  tmp = labels_indicator - x_soft ;
  dzdx = (tmp) ./ ((abs(tmp)-1) .* abs(tmp)) ; % Derivative of logloss
  dzdx(tmp == 0) = 0 ; % Remove inf
  dzdx(abs(tmp) == 1) = 0 ; % Remove inf
  dzdx = dzdx / (sz(1)*sz(2)) ;
  dzdx = x_soft .* bsxfun(@minus, dzdx, sum(dzdx .* x_soft, 3)) ; % Derivative of softmax
  dzdx(repmat(nans, 1, 1, sz(3), 1)) = 0 ; % Set derivatives to 0 where there were NaNs in the input or the ground truth

  assert(gather(all(~isnan(dzdx(:))))) ; % Debug check
  assert(gather(all(~isinf(dzdx(:))))) ; % Debug check
  out = dzdx ; % Return dzdx
end
