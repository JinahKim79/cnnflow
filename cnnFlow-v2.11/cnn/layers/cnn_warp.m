function X = cnn_warp(X, uvMap, referenceFrame, fillMode, dzdy)
%CNN_WARP  CNN Warping according to UV flow map

% Author: Damien Teney

[height, width, nFrames, nImages] = size(X) ;

if isempty(dzdy)
  % Forward
  if isempty(uvMap) || isequal(uvMap, 0), return ; end

  assert(size(uvMap, 1) == height) ;
  assert(size(uvMap, 2) == width) ;

  if isa(X, 'gpuArray')
    % Will make meshgrid() run on the GPU
    height = gpuArray(height) ;
    width = gpuArray(width) ;
  end

  [u, v] = meshgrid(1:width, 1:height) ;
  for i = 1:nImages
    tmp0 = X(:, :, referenceFrame, i) ;
    du = uvMap(:, :, 1, i) ;
    dv = uvMap(:, :, 2, i) ;
    % Warp each frame onto the reference one
    for c = 1:nFrames
      if c == referenceFrame, continue ; end % Nothing to do for the reference frame: skip it
      u2 = u + du * (c - referenceFrame) ; % Add 'du' for each frame
      v2 = v + dv * (c - referenceFrame) ;

      if fillMode == 4
        % Determine occlusions
        magnitudes = sqrt(du.^2 + dv.^2) ;
        u2b = round(u2) ; v2b = round(v2) ; u2b(u2b < 1) = 1 ; v2b(v2b < 1) = 1 ; u2b(u2b > width) = width ; v2b(v2b > height) = height ; % Replace out of range values
        is = v2b + (u2b-1)*height ; % Same as: is = sub2ind([width, height], v2, u2) ;
        tmpMax = accumarray([gather(v2b(:)), gather(u2b(:))], gather(magnitudes(:)), [gather(height), gather(width)], @max) ;
        tmpMax = imclose(tmpMax, strel('disk', 2, 0)) ; % Make it the *local* max
        invalid = magnitudes + 1 < tmpMax(is) ;
        tmpMax = imdilate(tmpMax, strel('disk', 2, 0)) ;
        %ims(invalid)

        tmp = X(:, :, c, i) ;
        %tmp(invalid) = NaN ;
        tmp2 = interp2(tmp, u2, v2, 'linear', NaN) ;
        tmp2(invalid) = NaN ;
        tmp2(isnan(tmp2)) = tmp0(isnan(tmp2)) ; % Replace NaNs by the values at the same location in the middle frame
        X(:, :, c, i) = tmp2 ;

      elseif fillMode == 2
        tmp2 = interp2(X(:, :, c, i), u2, v2, 'linear', NaN) ;
        if fillMode == 4, tmp2(becomingOccluded) = NaN ; end
        tmp2(isnan(tmp2)) = tmp0(isnan(tmp2)) ; % Replace NaNs by the values at the same location in the middle frame
        X(:, :, c, i) = tmp2 ;
      else
        u2(u2 < 1) = 1; u2(u2 > width) = width; % Fix out-of-image values
        v2(v2 < 1) = 1; v2(v2 > height) = height;
        %X(:, :, c, i) = interp2(X(:, :, c, i), u2, v2, 'cubic', 0) ; % Not supported on GPU
        X(:, :, c, i) = interp2(X(:, :, c, i), u2, v2, 'linear', 0) ;
      end
    end
  end

  if fillMode == 3
    % Replace invalid values by values from the original (unwarped) images (not perfect but better than blank values)
    invalidMask = (X == 0) ;
    X(invalidMask) = X(invalidMask) ;
  elseif fillMode == 0
    X(isnan(X)) = 0 ;
  elseif fillMode == -1
    % Nothing to do, keep NaN values
  end

else
  % Backward
  error('Not supported !') ;
end
