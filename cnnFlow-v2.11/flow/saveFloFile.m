function saveFloFile(img, fileName)
%SAVEfLOFILE Save optical flow in Middlebury benchmark format.

% Authors: Daniel Scharstein, Deqing Sun

[h, w, n] = size(img);

assert(n == 2);

fid = fopen(fileName, 'w');
if fid < 0
  error(['Cannot create file ' fileName]);
end

% Write header
tag = 'PIEH';
fwrite(fid, tag); 
fwrite(fid, w, 'int32');
fwrite(fid, h, 'int32');

% Write data
tmp = zeros(h, w*n);
tmp(:, (1:w)*n-1) = img(:,:,1);
tmp(:, (1:w)*n) = squeeze(img(:,:,2));
tmp = tmp';

fwrite(fid, tmp, 'float32');
fclose(fid);
