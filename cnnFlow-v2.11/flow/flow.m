function [finalLr, errorTr, errorVa, errorTe] = flow(varargin)
%FLOW Main function to call for CNN/optical flow experiments. Wrapper for flow_defineNetwork() and runCnn().

% Get parameters
p = flow_getParams({varargin{:}})

% Create network
net = feval(['flow_defineNetwork', num2str(p.modelId)], p) ;
net = initCnn(p, net) ;

% Run/train network
[net, info] = runCnn(p, net, @flow_measureError, @flow_processResults) ;

% Set output arguments
if nargout > 0
  finalLr = info{1}.lr(end) ;
  errorTr = info{1}.error(end, :) ;
  errorVa = info{2}.error(end, :) ;
  errorTe = info{3}.error(end, :) ;
end

% Display weights of the network
if p.displayWeights
  img = [] ; for i = 1:length(net), img = cat(5, img, makeOrientedFiltersImage(net{i}{6}.allFilters, 1, p.nOris) ); end, ims(img, false), % Motion filters of all scales
  img = [] ; for i = 1:length(net), img = cat(5, img, makeFiltersImage(net{i}{6}.filters) ); end, ims(img, false), % Motion filters of all scales (reduced set)
  img = [] ; for i = 1:length(net), img = cat(5, img, makeOrientedFiltersImage(net{i}{11}.allFilters, p.nOris, p.nOris) ); end, ims(img, false), % Smoothing filters of all scales
  img = [] ; for i = 1:length(net), img = cat(5, img, makeFiltersImage(net{i}{11}.filters) ); end, ims(img, false), % Smoothing filters of all scales (reduced set)
  makeOrientedFiltersImage(net{end}{14}.allFilters, p.nOris, p.nOris) ; % Decoding weights 1
  makeOrientedFiltersImage(net{end}{14}.filters, p.nOris/2+1, 1) ; % Decoding weights 1 (reduced set)
  %makeOrientedFiltersImage(net{end}{17}.filters, p.nOris, 1) ; % Weights of linear output layer (projection of all orientations onto UV components)
end
