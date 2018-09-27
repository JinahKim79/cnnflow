function h = subplottight(Ny, Nx, j, margin)
% Similar to subplot, but removes the spacing between axes.

if nargin < 4 
  margin = 0.1;
end

j = j-1;
x = mod(j,Nx)/Nx;
y = (Ny-fix(j/Nx)-1)/Ny;
h = axes('position', [x + margin/Nx ...
                      y + margin/Ny ...
                      1/Nx - 2*margin/Nx ...
                      1/Ny - 2*margin/Ny]);
