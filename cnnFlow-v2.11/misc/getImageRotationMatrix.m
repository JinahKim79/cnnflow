function result = getImageRotationMatrix(w, h, angle)
%GETIMAGEROTATIONMATRIX Transformation matrix to rotate a vectorized image.

% Author: Damien Teney

%%{
center = 1 + ([h, w] - 1) / 2;
R = [ +cos(angle) +sin(angle) ; -sin(angle) +cos(angle) ]';
result = zeros(h, w, h+2, w+2);
for jj = 1:w
  for ii = 1:h
    M = center + ([ii, jj] - center) * R; % Source indices
    %srcIndices = round(srcIndices * 100) / 100; % Round to 2 digits
    if any(M < 0) || any(M > [h w]+1)
      continue;
    else
      % Compute the relative areas with the 4 surrounding pixels
      % Triangle area mapping
      % Reference: http://www.leptonica.com/rotation.html
      %{
      C = ceil(M);
      F = floor(M);
      result(ii, jj, F(1), F(2)) = (C(2)-M(2)) * (C(1)-M(1)) ;
      result(ii, jj, F(1), C(2)) = (M(2)-F(2)) * (M(1)-F(1)) ;
      result(ii, jj, C(1), F(2)) = (C(2)-M(2)) * (M(1)-F(1)) ;
      result(ii, jj, C(1), C(2)) = (M(2)-F(2)) * (C(1)-M(1)) ;
      %}
      % Bilinear interpolation
      %%{
      F = floor(M);
      s = (M-F); t = (1-s);
      F = F + 1;
      C = F + 1;
      result(ii, jj, F(1), F(2)) = t(1) * t(2) ;
      result(ii, jj, F(1), C(2)) = t(1) * s(2) ;
      result(ii, jj, C(1), F(2)) = s(1) * t(2) ;
      result(ii, jj, C(1), C(2)) = s(1) * s(2) ;
      %}
    end
  end
end
result = reshape(result(:, :, 2:end-1, 2:end-1), h*w, h*w);
%}

%{
R = [ +cos(angle) +sin(angle) ; -sin(angle) +cos(angle) ]';

center = (h - 1) / 2;
[js, is] = meshgrid(1:h, 1:w);
dstIndices = [is(:), js(:)];
srcIndices = center + (dstIndices - center) * R;

%invalidIdx = any(srcIndices < 1, 2) | any(srcIndices(:, 1) > w, 2) | any(srcIndices(:, 2) > w, 2);
validIdx = all(srcIndices >= 1, 2) & all(srcIndices(:, 1) <= w, 2) & all(srcIndices(:, 2) <= w, 2);
srcIndices = srcIndices(validIdx, :);
dstIndices = dstIndices(validIdx, :);

% Get all 4 surrounding pixels
srcIndicesF = floor(srcIndices);
srcIndicesC = ceil(srcIndices);

result = zeros(h, w, h, w);
result(sub2ind([h, w, h, w], dstIndices(:, 1), dstIndices(:, 2), srcIndicesF(:, 1), srcIndicesF(:, 2))) = (srcIndicesC(:, 2)-srcIndices(:, 2) ) .* (srcIndicesC(:, 1)-srcIndices(:, 1) ) ;
result(sub2ind([h, w, h, w], dstIndices(:, 1), dstIndices(:, 2), srcIndicesF(:, 1), srcIndicesC(:, 2))) = (srcIndices(:, 2) -srcIndicesF(:, 2)) .* (srcIndices(:, 1) -srcIndicesF(:, 1)) ;
result(sub2ind([h, w, h, w], dstIndices(:, 1), dstIndices(:, 2), srcIndicesC(:, 1), srcIndicesF(:, 2))) = (srcIndicesC(:, 2)-srcIndices(:, 2) ) .* (srcIndices(:, 1) -srcIndicesF(:, 1)) ;
result(sub2ind([h, w, h, w], dstIndices(:, 1), dstIndices(:, 2), srcIndicesC(:, 1), srcIndicesC(:, 2))) = (srcIndices(:, 2) -srcIndicesF(:, 2)) .* (srcIndicesC(:, 1)-srcIndices(:, 1) ) ;
result = reshape(result, h*w, h*w) ;
%}
