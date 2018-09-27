function str = trimStr(str, maxLength)
%TRIMSTR Trim string to a given max length.
%   STR = TRIMSTR(STR, MAXLENGTH) shortens the given string so that is
%   has at most 'MAXLENGTH' characters.

% Author: Damien Teney

str = str(1:min(maxLength, length(str)));
