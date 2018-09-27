function flow_processResults(p, pred, gt, isRefinedResult, batchElement)
%FLOW_PROCESSRESULTS Make and save visualizations of ground truth (gt) and predicted (pred) flow maps.

% Find the scaling of the flow (max absolute magnitude)
persistent maxFlow ;
if ~all(isnan(gt(:))) % Valid ground truth available
  maxFlow = getMaxFlowMagnitude(gt) ; % Get max magnitude from ground truth
else
  if p.uniqueVisualizationMaxFlow % Use the same scale of flows for all results
    %if isempty(maxFlow) % First iteration
    if batchElement == 1 % First iteration
      maxFlow = p.uniqueVisualizationMaxFlow * getMaxFlowMagnitude(pred) ;
    end
  else % Get the max flow of each sequence
    maxFlow = getMaxFlowMagnitude(pred) ;
  end
end

% Get a base file name (in the results directory) matching the name of the input
fileName = p.files.fl(batchElement) ;
fileName = fileName{1} ; fileName = fileName{1} ;
fileName = strcat(p.expDir, filesep, fileName) ;
path = fileparts(fileName) ;
if ~exist(path, 'dir'), mkdir(path) ; end

% Save .flo file of predicted flow
saveFloFile(pred, strcat(fileName, '.flo')) ;

% Save image of ground truth flow
if ~all(isnan(gt(:))) % Valid ground truth is available
  saveFlowImage(gt, maxFlow, strcat(fileName, '-gt.flo.png'), p.visualizationQuiver) ;
end

% Save image of predicted flow
if isRefinedResult
  saveFlowImage(pred, maxFlow, strcat(fileName, '-refined.flo.png'), p.visualizationQuiver) ;
else
  saveFlowImage(pred, maxFlow, strcat(fileName, '.flo.png'), p.visualizationQuiver) ;
end

%==========================================================================
function saveFlowImage(uvMap, maxFlowMagnitude, fileName, quiverStride)

im = flowToColor(uvMap, maxFlowMagnitude);
figureId = figure ; imshow(im) ; drawnow() ; % Display the image

if quiverStride
  % Overlay arrows onto the flow map
  [x, y] = meshgrid(quiverStride : quiverStride : size(im, 2), quiverStride : quiverStride : size(im, 1)) ;
  uvMap = 0.8 * uvMap ./ maxFlowMagnitude ;
  %uvMap = bsxfun(@times, uvMap, max(abs(uvMap), [], 3) > 0.1) ; % Keep only values above a threshold
  u = uvMap(quiverStride : quiverStride : end, quiverStride : quiverStride : end, 1) ;
  v = uvMap(quiverStride : quiverStride : end, quiverStride : quiverStride : end, 2) ;
  valid = (abs(u) > 0) & (abs(v) > 0) ;
  hold on ;
  quiver(x(valid), y(valid), u(valid), v(valid), 'k-', 'LineWidth', 1.0, 'Color', [.4 .4 .4]) ;
  drawnow() ;
  set(figureId, 'Units', 'Inches'); pos = get(figureId, 'Position'); set(figureId, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', pos(3:4)) ; % Remove white borders before saving
  print(figureId, fileName, '-dpng', '-r0') ; % Save the figure (original resolution)
  %print(figureId, fileName, '-dpng') ; % Save the figure (display resolution)
else
  imwrite(im, fileName) ; % Save the image without the quiver (can use imwrite() instead of print(), it is faster)
end
