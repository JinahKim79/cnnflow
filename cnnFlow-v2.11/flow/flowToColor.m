function img = flowToColor(flow, maxMagnitude)
%FLOWTOCOLOR Make RGB image from UV flow map.
%  IMG = FLOWTOCOLOR(FLOW, MAXMAGNITUDE) color codes flow field, normalize
%  based on specified value.

%   Author: Deqing Sun; refactored by Damien Teney

assert(size(flow, 3) == 2);

u = flow(:, :, 1);
v = flow(:, :, 2);

% Fix unknown flow
idxUnknown = (abs(u)> 1e10) | (abs(v)> 1e9) ;
u(idxUnknown) = 0;
v(idxUnknown) = 0;

maxu = max(-999, max(u(:)));
minu = min(+999, min(u(:)));
maxv = max(-999, max(v(:)));
minv = min(+999, min(v(:)));

if nargin < 2
  maxMagnitude = getMaxFlowMagnitude(flow);
end

u = u / (maxMagnitude+eps);
v = v / (maxMagnitude+eps);

% Main code
nanIdx = isnan(u) | isnan(v);
u(nanIdx) = 0;
v(nanIdx) = 0;

colorwheel = makeColorwheel();
nColors = size(colorwheel, 1);

rad = sqrt(u.^2+v.^2);
a = atan2(-v, -u) / pi;
fk = (a+1) /2 * (nColors-1) + 1;  % -1~1 maped to 1~nColors
k0 = floor(fk);                   % 1, 2, ..., nColors
k1 = k0 + 1;
k1(k1 == nColors+1) = 1;
f = fk - k0;

for i = 1:size(colorwheel, 2)
  tmp = colorwheel(:, i);
  col0 = tmp(k0)/255;
  col1 = tmp(k1)/255;
  col = (1-f).*col0 + f.*col1; % Linear interpolation

  idx = (rad <= 1);   
  col(idx) = 1 - rad(idx).*(1-col(idx)); % Increase saturation with radius
  col(~idx) = col(~idx)*0.75; % Out of range

  img(:, :, i) = uint8(floor(255 * col.*(1-nanIdx)));
end

% Unknown flow
idxUnknown = repmat(idxUnknown, [1 1 3]);
img(idxUnknown) = 0;

%==========================================================================
function colorwheel = makeColorwheel()
% http://members.shaw.ca/quadibloc/other/colint.htm

RY = 15;
YG = 6;
GC = 4;
CB = 11;
BM = 13;
MR = 6;

nColors = RY + YG + GC + CB + BM + MR;

colorwheel = zeros(nColors, 3); % r g b

col = 0;
%RY
colorwheel(1:RY, 1) = 255;
colorwheel(1:RY, 2) = floor(255*(0:RY-1)/RY)';
col = col+RY;

%YG
colorwheel(col+(1:YG), 1) = 255 - floor(255*(0:YG-1)/YG)';
colorwheel(col+(1:YG), 2) = 255;
col = col+YG;

%GC
colorwheel(col+(1:GC), 2) = 255;
colorwheel(col+(1:GC), 3) = floor(255*(0:GC-1)/GC)';
col = col+GC;

%CB
colorwheel(col+(1:CB), 2) = 255 - floor(255*(0:CB-1)/CB)';
colorwheel(col+(1:CB), 3) = 255;
col = col+CB;

%BM
colorwheel(col+(1:BM), 3) = 255;
colorwheel(col+(1:BM), 1) = floor(255*(0:BM-1)/BM)';
col = col+BM;

%MR
colorwheel(col+(1:MR), 3) = 255 - floor(255*(0:MR-1)/MR)';
colorwheel(col+(1:MR), 1) = 255;
