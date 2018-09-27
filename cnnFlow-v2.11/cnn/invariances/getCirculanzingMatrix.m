function result = getCirculanzingMatrix(nChannelsIn, nChannelsOut, nOrientations, isCrossSectionSymmetric)
%GETTOEPLTZINGMATRIX Make transformation matrix to create filters with circulant structure from its cross-section or half cross-section (with isCrossSectionSymmetric=true).

% Author: Damien Teney

makeCirculantMatrix = @(v) ( toeplitz(v(:), v([1, end:-1:2])) ) ;
makeSymmetricCirculantMatrix = @(v) ( toeplitz(v([1:end, end-1:-1:2])) ) ; % Make a circulant and symmetric matrix

if isCrossSectionSymmetric
  nOrientationsSpecified = ceil(nOrientations/2)+1 ; % Symmetric cross-section: only half of the elements need to be specified
else
  nOrientationsSpecified = nOrientations ;
end

%result = zeros(nChannelsIn, nOrientations, nChannelsOut, nOrientations, nChannelsIn, nOrientationsSpecified, nChannelsOut) ;

for t = 1:nOrientationsSpecified
  % Make a 1-hot vector of size 'nOrientationsSpecified'
  v = zeros(nOrientationsSpecified, 1) ; v(t) = 1 ;

  if isCrossSectionSymmetric
    tmp = makeSymmetricCirculantMatrix(v) ;
  else
    tmp = makeCirculantMatrix(v) ;
  end

  for i = 1:nChannelsIn
    for o = 1:nChannelsOut
      %result(i, :, o, :, i, o, t) = tmp ;
      result(i, :, o, :, i, t, o) = tmp ;
    end
  end
end

result = reshape(result, nChannelsIn*nOrientations*nChannelsOut*nOrientations, nChannelsIn*nOrientationsSpecified*nChannelsOut) ;

% Demo of the use of the result of this function
%{
nOrientations = 8 ;
nOrientationsSpecified = ceil(nOrientations/2)+1 ;

% Filter 1x1, 1 input channel, 1 output channel
f = [5 3 2 1 0]' ;
f = reshape(f, nOrientationsSpecified, 1) ;
f2 = getCirculanzingMatrix(1, 1, nOrientations, true) * f ;
f2 = reshape(f2, 1, 1, nOrientations, nOrientations, []) ;
ims(f2)

% Filter 1x1, 1 input channel, 2 output channels
f = zeros(1, nOrientationsSpecified, 2, 1) ;
f(1, :, 1, 1) = [5 3 2 1 0] ;
f(1, :, 2, 1) = [5 5 -5 -5 0] ;
f = reshape(f, [], 1) ;
f2 = getCirculanzingMatrix(1, 2, nOrientations, true) * f ;
  f2 = reshape(f2, 1, nOrientations, 2, nOrientations) ;
  reshape(f2(1, :, 1, :), nOrientations, nOrientations) % Display weights for output channel 1
  reshape(f2(1, :, 2, :), nOrientations, nOrientations) % Display weights for output channel 2
f2 = reshape(f2, nOrientations, nOrientations*2) ;
makeFiltersImage(f2)

% Filter 1x1, 2 input channel, 2 output channels
f = zeros(2, nOrientationsSpecified, 2, 1) ;
f(1, :, 1, 1) = [5 3 2 1 0] ;
f(2, :, 2, 1) = [5 5 -5 -5 0] ;
makeFiltersImage(f)
f = reshape(f, [], 1) ;
U = getCirculanzingMatrix(2, 2, nOrientations, true) ;
f2 = U * f ;
  f2 = reshape(f2, 2, nOrientations, 2, nOrientations) ;
  reshape(f2(1, :, 1, :), nOrientations, nOrientations) % Display weights for input/output channel 1
  reshape(f2(2, :, 2, :), nOrientations, nOrientations) % Display weights for input/output channel 2
  reshape(f2(1, :, 2, :), nOrientations, nOrientations) % Display weights for input channel 1 / output channel 2 (all zeros)
f2 = reshape(f2, 1, 1, 2*nOrientations, 2*nOrientations) ;
makeFiltersImage(f2)

% Inverse
f2 = reshape(f2, [], 1) ;
f1 = pinv(U) * f2 ;
f  = reshape(f , 2, 2, nOrientationsSpecified) ; makeFiltersImage(f)  % Original
f1 = reshape(f1, 2, 2, nOrientationsSpecified) ; makeFiltersImage(f1) % Reconstructed: should be the same

nOrientations = 8 ;
nOrientationsSpecified = ceil(nOrientations/2)+1 ;

% Filter 1x1, 1 input channel, 1 output channel, NON-symmetric cross-section
nOrientations = 8 ;
nOrientationsSpecified = nOrientations ;
f = [1:8]' ;
f = reshape(f, nOrientationsSpecified, 1) ;
f2 = getCirculanzingMatrix(1, 1, nOrientations, false) * f ;
f2 = reshape(f2, 1, 1, nOrientations, nOrientations, []) ;
ims(f2)
%}
