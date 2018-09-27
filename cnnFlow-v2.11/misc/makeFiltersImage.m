function im = makeFiltersImage(f, rgbColorInput, doGroup, makeMovie, addDelimiter)
%MAKEFILTERSIMAGE Make visualization of filters (4D array).

% Author: Damien Teney

% Set default values for missing arguments
if nargin < 2
  rgbColorInput = false ;
end
if nargin < 3
  doGroup = true ;
end
if nargin < 4
  makeMovie = false ;
end
if nargin < 5
  addDelimiter = 1 * (size(f, 1) > 1) ; % Add delimiter only with kernels of size > 1x1
end

if ndims(f) == 5 % Several sets of filters for different recurrent iterations (concatenated along the 5th dimension)
  %displayWarning('Given filters are of size %d x %d x %d x %d x %d, displaying only 1st of the last dimension !\n', size(f, 1), size(f, 2), size(f, 3), size(f, 4), size(f, 5)) ;
  %f = f(:, :, :, :, 1) ;
  if makeMovie
    assert(nargout == 0) ;
    for i = 1:size(f, 5) % For each set of filters
      makeFiltersImage(f(:, :, :, :, i), rgbColorInput, doGroup, makeMovie, addDelimiter) ;
    end
  else
    im = [] ; % To accumulate the images of each set of filters
    for i = 1:size(f, 5) % For each set of filters
      im = cat(3, im, makeFiltersImage(f(:, :, :, :, i), rgbColorInput, doGroup, makeMovie, addDelimiter)) ; % Recursive call to this function for each set of filters
    end
    if nargout == 0
      ims(im, false) ;
      clear im ;
    end
  end
  return ;
end

if ndims(f) == 6
  sz = size(f) ;
  f = reshape(f, sz(1), sz(2), sz(3)*sz(4), sz(5)*sz(6)) ;
end

if (ndims(f) ~= 4) && (ndims(f) ~= 3)
  displayWarning('Given filters do not have the right number of dimensions !\n') ;
  if nargout > 0, im = [] ; end
  return ;
end

if isempty(f)
  displayWarning('Empty input !\n') ;
  if nargout > 0, im = [] ; end
  return ;
end

% Get values in [0,1]
%if all(f(:) >= 0) % Only positive values
%  f = f ./ max(abs(f(:))) ;
%else % Positive and negative values
  f = f ./ (eps+max(abs(f(:)))) ; f = (f / 2) + 0.5 ;
%end
%f = (f+1) / 2 ; % Map from [-1,+1] to [0,+1]

if addDelimiter
  % Will add delimiter between the filters after concatenation into a big image (the width of which is equal to the value of 'addDelimiter')
  f(end+1 : end+addDelimiter, :, :, :) = NaN ;
  f(:, end+1 : end+addDelimiter, :, :) = NaN ;
end

if size(f, 3) == 3 && rgbColorInput
  im = reshape(permute(f, [1 2 4 3]), size(f, 1), [], 3) ;
elseif size(f, 4) == 1
  im = reshape(f, size(f, 1), []) ;
elseif (size(f, 1) == 1) && (size(f, 2) == 1) % 1x1 kernels (pixelwise weights)
  im = squeeze(f) ;
else
  if doGroup % Given argument 'doGroup'
    if makeMovie % Given argument 'makeMovie'
      % Movie
      im = [] ;
      for i = 1:size(f, 4)
        tmp = [] ;
        for k = 1:size(f, 3)
          tmp = cat(3, tmp, f(:, :, k, i)) ;
        end
        im = cat(2, im, tmp) ;
      end
    else
      % Static image
      im = [] ;
      for i = 1:size(f, 4)
        tmp = [] ;
        for k = 1:size(f, 3)
          tmp = cat(1, tmp, f(:, :, k, i)) ;
        end
        im = cat(2, im, tmp) ;
      end
    end
  else
    im = reshape(f, size(f, 1), [], size(f, 3)) ; % Reshape to a wide image
    im = permute(im, [1 3 2]) ;
    im = reshape(im, size(im, 1) * size(im, 2), size(im, 3)) ;
  end
end

if addDelimiter
  im = im(1:end-addDelimiter, 1:end-addDelimiter, :, :) ; % Remove the last (bottom right of the image) delimiter
end

if nargout == 0
  if makeMovie % Given argument 'makeMovie'
    visualizeData3D_video(im, []) ; % Display the movie
  else
    im(isnan(im)) = 0 ;
    figure ; imshow(im, []) ; % Display the image
  end
  clear im ; % Do not return anything
end
