function outputStruct = matchImagesNoSeg(inputImagePath)

% SETUP VARS
debug = false;
terminalOutput = false;
numRepeats = 3;
useGCC = true;
maxFeatures = 500;
showUI = false;

% ALGORITHM START
load('Bill Images/goldenSiftResults.mat');

rawImage = imread(inputImagePath);
rawImage = imresize(rawImage, [700 NaN]);
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

% SIFT INPUT
[Fsamp,Dsamp] = vl_sift(single(sample), 'PeakThresh', 5);

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
matchSum = zeros(goldenRows,1);
Hs = cell(goldenRows,1);
for j = 1:goldenRows;
    
    clear H score ok ;
    
    Fgolden = goldenSiftResults{j, 5};
    Dgolden = goldenSiftResults{j, 6};
    
    [matches, scores]=vl_ubcmatch(Dsamp, Dgolden, 1.5);
    
    %sanity check: are there any matches? sometimes not.
    matchSum(j) = 0;
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
        
        %there is again another chance for matches to be empty, make sure
        %it's not
        
        if ~isempty(matches)
            matchSum(j) = sum(clearSelector);
            %GCC check
            if useGCC
                numMatches = size(matches,2) ;
                X1 = Fsamp(1:2,matches(1,:)) ; X1(3,:) = 1 ;
                X2 = Fgolden(1:2,matches(2,:)) ; X2(3,:) = 1 ;
                
                score = zeros(100,1);
                ok = cell(100,1);
                H = cell(100,1);
                for t = 1:100
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
                
                [score, best] = max(score) ;
                H = H{best} ;
                Hs{j} = H;
                ok = ok{best} ;
                matchSum(j) = sum(ok);
            end
        end
    end
    if terminalOutput
        fprintf('%d: matches %d, GCC matches %d, mean score %d\n', j, numel(matches)/2, matchSum(j), mean(scores))
    end
    
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

if showUI
    %map edges
    H = Hs{ind};
    xGolden = 10;
    yGolden = 10;
    
    pa = [xGolden; yGolden; 1];
    pb = H*pa;
    pb = pb/pb(3);
    xSample = pb(1)
    ySample = pb(2)
    
%     z =  H(3,1)*xGolden + H(3,2)*yGolden + H(3,3);
%     xSample = ( H(1,1)*xGolden + H(1,2)*yGolden + H(1,3) )/z
%     ySample = ( H(2,1)*xGolden + H(2,2)*yGolden + H(2,3) )/z
    imshow(rawImage)
    hold on
    plot(round(xSample),round(ySample),'o');
end

outputStruct.Country = { goldenSiftResults{ind, 1} };
outputStruct.ImgLoc = { goldenSiftResults{ind, 2} };
outputStruct.Features = { Dsamp };
outputStruct.matchSum = { matchSum };