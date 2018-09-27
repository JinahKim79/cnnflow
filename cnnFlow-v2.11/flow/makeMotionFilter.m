function [fEven, fOdd] = makeMotionFilter(nFrames, ori, speed, scale, widthScale, bidirectionalFlow, movingEnveloppe, squeezedEnveloppe)
%MAKEMOTIONFILTER Build spatiotemporal filter

%envelopeScale = 1.27 * scale ;
envelopeScale = 2.27 * scale ;
freqS = 0.25 / scale ; % In cycles/pixel
%freqS = 0.4 / scale ; % In cycles/pixel

r = 4 * widthScale ;
if bidirectionalFlow
  %[x, y, z] = meshgrid(-r:+r, -r:+r, -floor(nFrames/2) : +ceil(nFrames/2)-1) ; % Symmetrical temporal enveloppe
  [x, y, z] = meshgrid(-r:+r, -r:+r, 0 : (nFrames-1)) ; z = z(:, :, [[end : -1 : end-floor(nFrames/2)+1], [1:ceil(nFrames/2)]]) ; % Smaller enveloppe before the central frame than after
else
  [x, y, z] = meshgrid(-r:+r, -r:+r, 0 : (nFrames-1)) ;
end

% Spatial component
f1 = exp(j*2*pi*freqS * (cos(ori).*x + sin(ori).*y)) ;

% Spatial envelope
% Rotate enveloppe so that we can squeeze it (make it oval)
x2 = cos(-ori) .* x - sin(-ori) .* y ;
y2 = sin(-ori) .* x + cos(-ori) .* y ;
if movingEnveloppe
  x2 = x2 + z * speed ;
end
if squeezedEnveloppe
  f2 = exp(-(x2.^2 + y2.^2/16) / (2/16*envelopeScale^2)) ; % Deformed Gaussian
  %f2 = exp(-(x2.^2 + y2.^2) / (2*envelopeScale^2)) .* (abs(x2) < 1.5) ; % Hard threshold
else
  f2 = exp(-(x2.^2 + y2.^2) / (2*envelopeScale^2)) ;
end
f2(f2 < 1e-03) = 0 ;

% Temporal component
%freqT = 1 ; % In cycles/frame
%[0, 0.10, 0.15, 0.23]
%speed = freqT / freqS ;
%freqT = speed .* freqS / scale ;
freqT = speed .* freqS ;
f3 = exp(j*2*pi*freqT .* z) ;

% Temporal enveloppe
scaleT = 1.5 ;
f4 = exp(-abs(z) ./ scaleT) ;
f4(f4 < 1e-03) = 0 ;

% Combine everything
fEven = imag(f1) .* real(f3) + real(f1) .* imag(f3) ;
fEven = fEven .* f2 ; % Spatial enveloppe
fEven = fEven .* f4 ; % Temporal enveloppe
fOdd = real(f1) .* real(f3) - imag(f1) .* imag(f3) ;
fOdd = fOdd .* f2 ; % Spatial enveloppe
fOdd = fOdd .* f4 ; % Temporal enveloppe
