function X = cnn_downsize(X, scaleFactor, dzdy)
%CNN_DOWNSIZE  CNN Image smoothing and bicubic downsampling

% Author: Damien Teney

if isempty(dzdy)
  % Forward
  if scaleFactor == 1, return ; end % Nothing to do
  assert(scaleFactor < 1) ;

  [h, w, c, n] = size(X) ;

  % Gaussian smoothing
  sigma = 1 / sqrt(2*scaleFactor) ;
  kernel = fspecial('gauss', [99, 99], sigma) ;
  a = diag(kernel) ;
  cut = find(a > 0.0001, 1, 'first') ;
  kernel = kernel(cut:100-cut, cut:100-cut) ;
  kernel = kernel/sum(kernel(:)) ;
  if length(kernel) > 1
    X = imfilter(X, kernel, 'replicate') ;
  end

  % Resize
  X = reshape(X, h, w, c*n) ; % Collapse the last 2 dimensions to process everything at once
  X = imresize(X, scaleFactor, 'bicubic') ;
  %{
  if isa(X, 'gpuArray')
    X = gpuArray(imresize(gather(X), scaleFactor, 'bilinear')) ; % Bilinear method not supported on the GPU
  else
    X = imresize(X, scaleFactor, 'bilinear') ;
  end
  %}
  X = reshape(X, size(X, 1), size(X, 2), c, n) ; % Restore the last 2 dimensions

else
  % Backward
  error('Not supported') ;
end
