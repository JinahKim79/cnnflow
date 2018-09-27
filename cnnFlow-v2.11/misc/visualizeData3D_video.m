function visualizeData3D_video(data, range)
%VISUALIZEDATA3D_VIDEO Display 3D data as a looping sequence over all slices along the 3rd dimension.

% Author: Damien Teney

assert(ndims(data) == 3);

data = double(data);

warning('off', 'Images:initSize:adjustingMag');

if nargin < 2 || isempty(range)
  % Range of values not given: automatic selection
  range = [min(data(:)) max(data(:))];
  if range(2) == range(1), range = [0 1]; end % Special case (all values == 0)
end

figureId = gcf;
clf(figureId, 'reset');
set(figureId, 'currentch', '0'); % Clear the property 'CurrentCharacter'
axisId = gca;

if size(data, 1) < 150 || size(data, 2) < 150
  zoomLevel = 400; % In percents
else
  zoomLevel = 100; % In percents
end

nSlices = size(data, 3);
while true
  %for k = 1:nSlices % Loop through slices forward then back at the beginning
  for k = [ 1:nSlices  nSlices-1:-1:2 ] % Go back a forth
    imshow(data(:, :, k), range, 'InitialMagnification', zoomLevel, 'Parent', axisId);
    drawnow();
    pause(1 / 25);

    % Check for a key pressed by the user
    if ~isempty(get(figureId, 'CurrentCharacter')) && get(figureId, 'CurrentCharacter') ~= '0'
      close(figureId);
      return;
    end
  end
end
