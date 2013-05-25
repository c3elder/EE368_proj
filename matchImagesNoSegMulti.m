function outputStruct = matchImagesNoSegMulti(inputImagePath)

% SETUP VARS
debug = false;
numRepeats = 3;
maxFeatures = 500;
numRansacIterations = 100;
showUI = true;
postGCCMatchThreshold = 10;
peakThreshold = 5;
textSize = 18;
textBorder = 7;

% ALGORITHM START
load('Bill Images/goldenSiftResults.mat');

rawImage = imread(inputImagePath);
rawImage = imresize(rawImage, [800 NaN]);
if debug %store image for showing later
    im1 = rawImage;
end

[M N C] = size(rawImage);
if C==3
    sample = rgb2gray(rawImage);
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

while billsToBeFound
    % SETUP
    matchSum = zeros(goldenRows,1);
    matchSumOrig = zeros(goldenRows,1);
    Hs = cell(goldenRows,1);
    matchesArrayOrig = cell(goldenRows,1);
    matchesArray = cell(goldenRows,1);
    
    % GET INITIAL MATCHES
    for j = 1:goldenRows;
        
        Fgolden = goldenSiftResults{j, 5};
        Dgolden = goldenSiftResults{j, 6};
        
        [matches, scores]=vl_ubcmatch(Dsamp, Dgolden, 1.5);
        
        %sanity check: are there any matches? sometimes not.
        matchSum(j) = 0;
        matchSumOrig(j) = 0;
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
            scores = scores(:, clearSelector);
        end
        %there is again another chance for matches to be empty, make sure it's not
        if ~isempty(matches)
            numMatches = size(matches,2) ;
            matchSum(j) = numMatches;
            matchSumOrig(j) = numMatches;
        end
        matchesArrayOrig{j} = matches;

        %matches = matchesArrayOrig{j};
        if ~isempty(matches)
            numMatches = size(matches,2) ;
            
            %GCC check
            X1 = Fsamp(1:2,matches(1,:)) ; X1(3,:) = 1 ;
            X2 = Fgolden(1:2,matches(2,:)) ; X2(3,:) = 1 ;
            
            score = zeros(numRansacIterations,1);
            ok = cell(numRansacIterations,1);
            H = cell(numRansacIterations,1);
            for t = 1:numRansacIterations
                % estimate homograpyh
                subset = vl_colsubset(1:numMatches, 4) ;
                A = [] ;
                for i = subset
                    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
                end
                [U,S,V] = svd(A);
                H{t} = reshape(V(:,9),3,3) ;
                
                % score homography
                X2_ = H{t} * X1 ;
                du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
                dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
                ok{t} = (du.*du + dv.*dv) < 6*6 ;
                score(t) = sum(ok{t}) ;
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
    val
    if val>postGCCMatchThreshold        
        %keep the relevant results
        outputStruct.goldenIndex(end+1) = ind;
        outputStruct.tformMatrix = [outputStruct.tformMatrix, { Hs{ind} }];
        outputStruct.Country = [outputStruct.Country, { goldenSiftResults{ind, 1} }];
        outputStruct.ImgLoc = [outputStruct.ImgLoc, { goldenSiftResults{ind, 2} }];
        outputStruct.Features = [outputStruct.Features, { Dsamp }];
        outputStruct.matchSum = [outputStruct.matchSum, { matchSum }];
        outputStruct.matchSumOrig = [outputStruct.matchSumOrig,  { matchSumOrig }];
             
        %remove the matched features from Dsamp and Fsamp entirely
        matches = matchesArray{ind};
        Fsamp(:, matches(1,:)) = [];
        Dsamp(:, matches(1,:)) = [];
    else
        % no more bill can be found, kill the engine
        billsToBeFound = false;
    end
end

%match the corners
if showUI
    figure
    imshow(rawImage)
    hold on
    
    for j = 1:length(outputStruct.goldenIndex)
        ind = outputStruct.goldenIndex(j);
        H = outputStruct.tformMatrix{j};
        imR = goldenSiftResults{ind, 8};
        imC = goldenSiftResults{ind, 9};
        
        invH = inv(H);
        border = 5;

        pa = zeros(2,4);
        pa(:,1) = [-border; -border];
        pa(:,2) = [-border; imR+border];
        pa(:,3) = [imC+border; imR+border];
        pa(:,4) = [imC+border; -border];

        
        pa = [pa; ones(1, numel(pa)/2)];
        Cx = 0;
        Cy = 0;
        for i=1:numel(pa)/3
            paInUse = pa(:,i);
            pb = invH*paInUse;
            pb = pb/pb(3);
            xSample = pb(1);
            Cx = Cx + xSample;
            ySample = pb(2);
            Cy = Cy + ySample;
            plot(round(xSample),round(ySample),'*k');
        end
        Cx = round(Cx/4);
        Cy = round(Cy/4);
        plot(Cx, Cy, 'xr')
        
        createText(Cx,Cy,[goldenSiftResults{ind, 1}], textSize);
        createText(Cx,Cy+textSize+textBorder,[num2str(goldenSiftResults{ind, 3})], textSize);
        
    end

end

end

function createText(Cx,Cy, inpString, textSize)
text(Cx+1,Cy+1,inpString, 'HorizontalAlignment','center', 'FontSize', textSize, 'Color', [0 0 0])
text(Cx-1,Cy-1,inpString, 'HorizontalAlignment','center', 'FontSize', textSize, 'Color', [0 0 0])
text(Cx+1,Cy-1,inpString, 'HorizontalAlignment','center', 'FontSize', textSize, 'Color', [0 0 0])
text(Cx-1,Cy+1,inpString, 'HorizontalAlignment','center', 'FontSize', textSize, 'Color', [0 0 0])
text(Cx,Cy,inpString, 'HorizontalAlignment','center', 'FontSize', textSize, 'Color', [1 1 1])
end