% Pack 3 grayscale images in 1 file as the separate color channels

% Author: Damien Teney

% Load 3 RGB images
tmp1 = imread('frame_0009.png') ;
tmp2 = imread('frame_0010.png') ;
tmp3 = imread('frame_0011.png') ;

% Convert them to grayscale and concatenate along the 3rd dimension
img = cat(3, rgb2gray(tmp1), rgb2gray(tmp2), rgb2gray(tmp3)) ;

% Save as 1 RGB image, with each grayscale frame as one channel
img = single(img) ./ 255 ;
imwrite(img, 'frames.png') ;
