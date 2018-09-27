function ims(x, displayInfo)
%IMS Display data in various forms, useful for debug.

%  Author: Damien Teney

if nargin < 2 % Missing argument
  displayInfo = @fprintf; % Set default value
elseif ~displayInfo
  displayInfo = @(varargin)([]);
end

if isa(x, 'gpuArray')
  displayInfo('GPU array\n');
end

x = gather(x);
if issparse(x)
  displayInfo('Sparse input\n');
  x = full(x);
end

displayInfo('Size:'); displayInfo('  %u', size(x)); displayInfo('\n');
if numel(x) == 0, return ; end

x = squeeze(x);

if length(x) == numel(x) % 1-dimensional vector
  figure;
  x2 = real(x);
  x2 = x; x2(isnan(x2)) = min(x2(:));
  if length(x2) > 1000
    hist(double(x2)); title('Histogram (distribution of values)') ;
  else
    bar(double(x2));
  end
elseif ndims(x) == 2 % 2D matrix/image
  figure;
  x2 = real(x);
  x2(isnan(x2)) = min(x2(:));
  imshow(x2, []);
%{
elseif (ndims(x) == 3) && (size(x, 3) == 3)) % 3-channel image
  figure;
  x2 = x; x2(isnan(x2)) = min(x2(:));
  imshow(x2, []);
%}
elseif ndims(x) == 3
  figure;
  x2 = real(x);
  x2(isnan(x2)) = min(x2(:));
  visualizeData3D_slider(x2);
elseif ndims(x) >= 3
  x2 = x(:, :, :, 1);
  displayInfo('Display first 3 dimensions:\t'); disp(size(x2));
  figure;
  x2 = real(x2);
  x2(isnan(x2)) = min(x2(:));
  visualizeData3D_slider(x2);
else
  %for i= 1:ndims(x), displayInfo('%d ', size(x, i)); end
  %displayInfo('\n');
end

displayInfo('Range:\t[%f   %f]\t%s', min(x(:)), max(x(:)), class(x));
if any(isnan(x(:)))
  displayInfo('\twith NaNs');
end
if any(~isreal(x(:)))
  displayInfo('\twith imaginary numbers');
end
displayInfo('\n');
