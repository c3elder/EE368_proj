function outputStruct = matchImagesMultiOptimized(inputImagePath, output_img_path)

%run('vlfeat-0.9.16\toolbox\vl_setup');

if nargin < 2
    output_img_path = 'C:\xampp-portable\htdocs\output\test.jpg';
end

tic

% SETUP VARS

numRepeats = 1;
maxFeatures = 3000;
numRansacIterations = 400;
ubcThresh = 2.0;
postGCCMatchThreshold = 10;
GCCNumRandFeatures = 4;
peakThreshold = 0.1;
earlyTermThresh = 0.2;
gccScoreThreshold = 30;
centroidGap = 80;

debug = false;
showUI = true;
droidCamera = true;
textSize = 30;
textBorder = 12;

% ALGORITHM START
load('Bill Images/goldenSiftResults.mat');

rawImage = imread(inputImagePath, 'jpg');
if droidCamera
   rawImage = rawImage(1:2:end,1:2:end,:);
else
    rawImage = rawImage(1:3:end,1:3:end,:);
    %rawImage = imresize(rawImage, [800 NaN]);
end

if debug %store image for showing later
    im1 = rawImage;
end

[M N C] = size(rawImage);
if C==3
    sample = rgb2gray(rawImage);
    %sample = rgb2ycbcr(rawImage);
    %sample = sample(:,:,1);
else
    sample = rawImage;
end

outputStruct = struct();
outputStruct.Country = {};
outputStruct.ImgLoc = {};
outputStruct.Features = {};
outputStruct.matchSum = {};
outputStruct.matchSumOrig = {};
outputStruct.tformMatrix = {};
outputStruct.goldenIndex = [];
outputStruct.centroid = [];
outputStruct.border = {};

% SIFT INPUT
[Fsamp,Dsamp] = vl_sift(single(sample), 'PeakThresh', peakThreshold);

[Rsamp, Csamp] = size(Fsamp);
if Csamp < maxFeatures
    maxFeatures = Csamp;
end

% LIMIT TO TOP maxFeatures
[sortedList, sortedInd ] = sort(Fsamp(1,:), 'descend');
Fsamp = Fsamp(:, sortedInd(1:maxFeatures));
Dsamp = Dsamp(:, sortedInd(1:maxFeatures));
Dsamp = single(Dsamp);

[goldenRows, goldenCols] = size(goldenSiftResults);

billsToBeFound = true;

matchesArrayOrig = cell(goldenRows,1);
matchesArray = cell(goldenRows,1);

%Get the matches in parallel
parfor j=1:goldenRows;
    Dgolden = goldenSiftResults{j, 6};
    [matches, scores]=vl_ubcmatch(Dsamp, Dgolden, ubcThresh);
    
    if ~isempty(matches)
        %for some unknown reason, tons of features end up mapping to the same
        %point in the golden bill, remove repeats
        uniqueValues = unique(matches(2,:));
        uniqueValueSums = zeros(length(uniqueValues), 1);
        for k = 1:length(uniqueValues)
            uniqueValueSums(k) = sum(matches(2,:)==uniqueValues(k));
        end
        repeatedValues = uniqueValues(uniqueValueSums>numRepeats); %if more than numRepeats of same descriptor
        clearSelector = true(1, length(matches(2,:)));
        for k=1:length(repeatedValues)
            clearSelector = clearSelector & (matches(2, :) ~= repeatedValues(k));
        end
        matches = matches(:, clearSelector);
        %scores = scores(:, clearSelector);
    end
    
    matchesArrayOrig{j} = matches;
end

%Iterative to find the bills
while billsToBeFound
    % SETUP
    matchSum = zeros(goldenRows,1);
    matchSumOrig = zeros(goldenRows,1);
    Hs = cell(goldenRows,1);

    % GET INITIAL MATCHES
    for j = 1:goldenRows;
        
        Fgolden = goldenSiftResults{j, 5};
        Dgolden = goldenSiftResults{j, 6};
        
        matchSum(j) = 0;
        matchSumOrig(j) = 0;
        matches = matchesArrayOrig{j}; %extract from earlier run
        
        %there is again another chance for matches to be empty, make sure it's not
        if ~isempty(matches)
            numMatches = size(matches,2) ;
            matchSum(j) = numMatches;
            matchSumOrig(j) = numMatches;
        end
        
        %matches = matchesArrayOrig{j};
        if ~isempty(matches) && numMatches > GCCNumRandFeatures
            numMatches = size(matches,2) ;
            
            %GCC check
            X1 = Fgolden(1:2,matches(2,:)) ; X1(3,:) = 1 ;
            X2 = Fsamp(1:2,matches(1,:)) ; X2(3,:) = 1 ;
            
            
            score = zeros(numRansacIterations,1);
            ok = cell(numRansacIterations,1);
            H = cell(numRansacIterations,1);
            
            %pull geometry for GCC sanity check
            border = 5;
            imR = goldenSiftResults{j, 8};
            imC = goldenSiftResults{j, 9};
            pa = zeros(2,4);
            pa(:,1) = [-border; -border];
            pa(:,2) = [-border; imR+border];
            pa(:,3) = [imC+border; imR+border];
            pa(:,4) = [imC+border; -border];
            pa = [pa; ones(1, numel(pa)/2)];
            centroids = outputStruct.centroid;
            %run ransac in parallel
            parfor t = 1:numRansacIterations
                %subset = vl_colsubset(1:numMatches, 4) ;
                subset = randperm(numMatches,GCCNumRandFeatures);
                A = [];
                for i=subset
                    A(end+1:end+2, :) = [X1(1,i), 0, -X1(1,i)*X2(1,i), X1(2,i), 0, -X1(2,i)*X2(1,i), 1, 0 -X2(1,i); ...
                                         0, X1(1,i), -X1(1,i)*X2(2,i), 0, X1(2,i), -X1(2,i)*X2(2,i), 0, 1, -X2(2,i)];
                end
                [U,S,V] = svd(A);
                theta = V(:,9);
                H{t} = reshape(theta,3,3);
                
                % score homography
                X2_ = H{t} * X1 ;
                du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
                dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
                ok{t} = (du.*du + dv.*dv) < gccScoreThreshold ;
                score(t) = sum(ok{t}) ;
                
                % sanity check: if H produces a centroid in the middle of a
                % previously found bill, kill it
                if ~isempty(centroids)
                    xSample = zeros(1,numel(pa)/3);
                    ySample = zeros(1,numel(pa)/3);
                    for i=1:numel(pa)/3
                        paInUse = pa(:,i);
                        pb = H{t}*paInUse;
                        pb = pb/pb(3);
                        xSample(i) = pb(1);
                        ySample(i) = pb(2);
                    end
                    Cx = round(mean(xSample));
                    Cy = round(mean(ySample));
                    if any( (abs(centroids(1,:)-Cx)<centroidGap) & (abs(centroids(2,:)-Cy)<centroidGap) )
                      % fprintf('centroid in close proximity to another\n');
                      score(t) = 0;
                    end
                end
                
                
            end
            
            [valScores, best] = max(score) ;
            H = H{best} ;
            Hs{j} = H; %store for future calculations
            ok = ok{best} ;
            matchesArray{j} = matches(1:2,ok);
            matchSum(j) = sum(ok);
        end
        %show all the matching points found with GCC if debug is on
        if debug
            if ~isempty(matches)
                f1 = Fsamp;
                f2 = Fgolden;
                
                goldenDir = ['Bill Images/' goldenSiftResults{j, 1} '/' goldenSiftResults{j, 2}];
                im2 = imread(goldenDir);
                
                dh1 = max(size(im2,1)-size(im1,1),0) ;
                dh2 = max(size(im1,1)-size(im2,1),0) ;
                
                figure; clf ;
                subplot(2,1,1) ;
                imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
                o = size(im1,2) ;
                line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
                    [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
                title(sprintf('%d tentative matches', numMatches)) ;
                axis image off ;
                
                subplot(2,1,2) ;
                imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
                o = size(im1,2) ;
                line([f1(1,matches(1,ok));f2(1,matches(2,ok))+o], ...
                    [f1(2,matches(1,ok));f2(2,matches(2,ok))]) ;
                title(sprintf('After GCC: %d (%.2f%%) inliner matches out of %d', ...
                    sum(ok), ...
                    100*sum(ok)/numMatches, ...
                    numMatches)) ;
                axis image off ;
                
                drawnow ;
            else
                fprintf('DEBUG: Figure %d not shown because there are 0 matches\n', j);
            end
        end
        
    end
        
    [val, ind] = max(matchSum);
    %final sanity check
    
    if val>postGCCMatchThreshold        
        %calculate the centroid and border using H
        border = 5;
        imR = goldenSiftResults{ind, 8};
        imC = goldenSiftResults{ind, 9};
        pa = zeros(2,4);
        pa(:,1) = [-border; -border];
        pa(:,2) = [-border; imR+border];
        pa(:,3) = [imC+border; imR+border];
        pa(:,4) = [imC+border; -border];
        H = Hs{ind};
        
        pa = [pa; ones(1, numel(pa)/2)];
        Cx = 0;
        Cy = 0;
        xSample = zeros(1,numel(pa)/3);
        ySample = zeros(1,numel(pa)/3);
        for i=1:numel(pa)/3
            paInUse = pa(:,i);
            pb = H*paInUse;
            pb = pb/pb(3);
            xSample(i) = pb(1);
            ySample(i) = pb(2);
        end
        Cx = round(mean(xSample));
        Cy = round(mean(ySample));
        
        %keep the relevant results
        outputStruct.centroid(1:2, end+1) = [Cx; Cy];
        outputStruct.border{end+1} = [xSample; ySample];
        outputStruct.goldenIndex(end+1) = ind;
        outputStruct.tformMatrix = [outputStruct.tformMatrix, { Hs{ind} }];
        outputStruct.Country = [outputStruct.Country, { goldenSiftResults{ind, 1} }];
        outputStruct.ImgLoc = [outputStruct.ImgLoc, { goldenSiftResults{ind, 2} }];
        outputStruct.Features = [outputStruct.Features, { Dsamp }];
        outputStruct.matchSum = [outputStruct.matchSum, { matchSum }];
        outputStruct.matchSumOrig = [outputStruct.matchSumOrig,  { matchSumOrig }];
             
        %remove the matched features from Dsamp and Fsamp entirely
        matches = matchesArray{ind};
        
        matchSumOrigNext = zeros(goldenRows,1);
        for j=1:goldenRows
            oldMatches = matchesArrayOrig{j};
            sel = oldMatches(1,:);
            oldMatches(:, ismember(sel,matches(1,:))) = [];
            matchesArrayOrig{j} = oldMatches;
            matchSumOrigNext(j) = size(oldMatches,2);
        end
        matchSumRatio = max(matchSumOrigNext)/max(matchSumOrig);
        if  matchSumRatio < earlyTermThresh
        %the next round will have a matchSumOrig = a if a/this round's
        %matchSumOrig < thresh use early termination
            billsToBeFound = false;
            fprintf(' - matchSumRatio %0.2f (next: %d, current: %d) - Early Termination\n', ...
                    matchSumRatio, max(matchSumOrigNext), max(matchSumOrig))
        else
            fprintf(' - matchSumRatio %0.2f (next: %d, current: %d)\n', ...
                    matchSumRatio, max(matchSumOrigNext), max(matchSumOrig))
        end
    else
        % no more bill can be found, kill the engine
        billsToBeFound = false;
        fprintf(' - max(matchedSum)=%d < postGCCMatchThreshold=%d - Natural Termination\n', val, postGCCMatchThreshold)
    end
end

toc

%match the corners
if showUI
    figure
    imshow(rawImage)
    hold on
    
    grandTotal = 0;
    for j = 1:length(outputStruct.goldenIndex)
        ind = outputStruct.goldenIndex(j);
        
        Cx = outputStruct.centroid(1,j);
        Cy = outputStruct.centroid(2,j);
        border = outputStruct.border{j};
        plot(round(border(1,:)),round(border(2,:)),'*k');
        plot(Cx, Cy, 'xr')
        
        nativeBillVal = goldenSiftResults{ind, 3};
        USDBillVal = nativeBillVal*goldenSiftResults{ind, 7};
        
        [subSampImgR subSampImgC] = size(sample);
        
        textEdgeBarrier = 80;
        if Cx < textEdgeBarrier
            Cx = textEdgeBarrier;
        elseif Cx > subSampImgC - textEdgeBarrier
            Cx = subSampImgC - textEdgeBarrier;
        end
        if Cy < textEdgeBarrier
            Cy = textEdgeBarrier;
        elseif Cy > subSampImgR - textEdgeBarrier
            Cy = subSampImgR - textEdgeBarrier;
        end
        
        createText(Cx,Cy,[goldenSiftResults{ind, 1} ' ' num2str(nativeBillVal)], textSize-5, 'center');
        createText(Cx,Cy+textSize+textBorder,sprintf('$%.2f', USDBillVal), textSize, 'center');
        grandTotal = grandTotal + USDBillVal;
    end
    createText(50,50,sprintf('Total $%.2f', grandTotal), textSize-5, 'left');
    
    f = getframe(gca);
    im = frame2im(f);
    imwrite(im,output_img_path, 'jpg');

end

end

function createText(Cx,Cy, inpString, textSize, alignTo)
text(Cx+1,Cy+1,inpString, 'HorizontalAlignment',alignTo, 'FontSize', textSize, 'Color', [0 0 0])
text(Cx-1,Cy-1,inpString, 'HorizontalAlignment',alignTo, 'FontSize', textSize, 'Color', [0 0 0])
text(Cx+1,Cy-1,inpString, 'HorizontalAlignment',alignTo, 'FontSize', textSize, 'Color', [0 0 0])
text(Cx-1,Cy+1,inpString, 'HorizontalAlignment',alignTo, 'FontSize', textSize, 'Color', [0 0 0])
text(Cx,Cy,inpString, 'HorizontalAlignment', alignTo, 'FontSize', textSize, 'Color', 'y')
end

function [estX1, estX2] = homographEstimation(th, x1, x2)
    bot = th(3)*x1 + th(6)*x2 + th(9);
    estX1 = (th(1)*x1 + th(4)*x2 + th(7))/bot;
    estX2 = (th(2)*x1 + th(5)*x2 + th(8))/bot;
end