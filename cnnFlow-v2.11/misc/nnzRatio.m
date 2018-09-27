function r = nnzRatio(m)
%NNZRATIO Ratio of non-zero elements in an array.
%   R = NNZRATIO(M) returns the ratio of non-zero elements in the array M.

%   Author: Damien Teney

r = nnz(m(:)) / numel(m);
