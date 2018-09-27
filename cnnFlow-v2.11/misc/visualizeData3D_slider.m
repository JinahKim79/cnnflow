function visualizeData3D_slider(data, range)
%VISUALIZEDATA3D_SLIDER displays 3D data one slice (along the 3rd dimension) at a time, with a slider to select other slices.

% Author: Damien Teney

  if isempty(data)
    return;
  end

  if all(isnan(data(:)))
    fprintf('All data is NaNs !\n');
    return ;
  end

  if ndims(data) ~= 3
    fprintf('Input data is not 3D !\n');
    disp(size(data));
  end
  if (ndims(data) ~= 3) && (ndims(data) ~= 2)
    return;
  end

  data = double(data);

  if nargin < 2 || isempty(range)
    % Range of values not given: automatic selection
    range = [min(data(:)) max(data(:))];
    if range(2) == range(1), range = [0 1]; end % Special case (all values == 0)
  end

  nSlices = size(data, 3);
  sliceId = ceil((size(data, 3) + 1) / 2); % Default slice: middle one

  figureId = gcf;
  clf(figureId, 'reset');
  figure(figureId); 

  % Add a slider
  if ndims(data) == 2
    sliderId = uicontrol('Style', 'slider', 'Min', 1, 'Max', 2, 'Value', 1, 'SliderStep', [1 1], 'Position', [5 5 200 25], ...
      'Callback', {@changeSlice, figureId, data, range});
  elseif ndims(data) == 3
    step = [1 1] / (nSlices - 1);
    sliderId = uicontrol('Style', 'slider', 'Min', 1, 'Max', nSlices, 'Value', sliceId, ...
      'SliderStep', step, 'Position', [5 5 200 25], ...
      'Callback', {@changeSlice, figureId, data, range});
    uicontrol(sliderId); % Give focus to the slider so that we can use keyboard arrows to change the slide
  end
  changeSlice(sliderId, [], figureId, data, range);
end

function changeSlice(src, ~, figureId, data, range)
  % Callback function for the slider: display another slice of the data
  if ndims(data) == 3
    sliceId = round(get(src, 'Value'));
  else
    sliceId = 1;
  end

  figure(figureId); imshow(data(:, :, sliceId), range);
  %title(sprintf('Slice %d / %d', sliceId, size(data, 3)));
  title(sprintf('Slice %d / %d   Dimensions: %ux%u \n Range: %.2f..%.2f   Integral: %.2f', ...
    sliceId, size(data, 3), ...
    size(data, 1), size(data, 2), ...
    min(data(:)), max(data(:)), ...
    sum(data(:))));
  axis off;
  drawnow();
end
