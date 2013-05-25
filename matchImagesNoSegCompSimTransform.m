function outputStruct = matchImagesNoSegCompSimTransform(inputImagePath)

% SETUP VARS
debug = false;
terminalOutput = false;
numRepeats = 3;
useGCC = true;
maxFeatures = 500;
showUI = true;

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
matchesCell = cell(goldenRows, 1);
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
                for t = 1:400
                    % estimate homograpyh
                    subset = vl_colsubset(1:numMatches, 4) ;
                    A = [] ;
                    for i = subset
                        A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
                    end
                    [U,S,V] = svd(A);
                    H{t} = reshape(V(:,9),3,3) ;
                    
                    %tform = cp2tform(Fgolden(1:2,matches(1,subset))', Fsamp(1:2,matches(1,subset))', 'projective');
                    %fprintf('worked')
                    %[Fsamp(1:2,matches(1,2))' 1]*tform.tdata.T;
                    %H{t} = tform.tdata.T;
                    
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

%once we know the right bill, get the best estimate of the edge positions
%possible

%run gcc again
Fgolden = goldenSiftResults{ind, 5};
Dgolden = goldenSiftResults{ind, 6};

[matches, scores]=vl_ubcmatch(Dsamp, Dgolden, 1.5);
numMatches = size(matches,2) ;

T = cell(400,1);
C = cell(400,1);
ok = cell(400,1);

X1all = Fgolden(1:2,matches(2,:));
X2all = Fsamp(1:2,matches(1,:));

for t = 1:400
    %estimate similarity transform
    %optimizations from http://vision.ece.ucsb.edu/~zuliani/Research/RANSAC/docs/RANSAC4Dummies.pdf
    subset = vl_colsubset(1:numMatches, 2) ;
    X2 = Fsamp(1:2,matches(1,subset));
    X1 = Fgolden(1:2,matches(2,subset));
    
    MM = X1(:,1) - X1(:,2);
    detMM = MM(1)*MM(1) + MM(2)*MM(2);
    MMi = MM/detMM;
    Delta = X2(:,1) - X2(:,2);
    
    theta = zeros(1,4);
    theta(1) = MMi(1)*Delta(1) + MMi(2)*Delta(2);
    theta(2) = MMi(1)*Delta(2) - MMi(2)*Delta(1);
    theta(3) = X2(1,2) - theta(1)*X1(1,2) + theta(2)*X1(2,2);
    theta(4) = X2(2,2) - theta(1)*X1(2,2) - theta(2)*X1(1,2);
       
    T{t} = [theta(1), -theta(2); theta(2), theta(1)];
    C{t} = [theta(3); theta(4)];
    
    %Score the results
    X2allEst = T{t}*X1all + [C{t}(1)*ones(1, numMatches); C{t}(2)*ones(1, numMatches)];
    
    du = X2allEst(1,:) - X2all(1,:);
    dv = X2allEst(2,:) - X2all(2,:);
    ok{t} = (du.*du + dv.*dv) < 6*6 ;
    score(t) = sum(ok{t});    
end

[val best] = max(score);

T = T{best};
C = C{best};

if showUI
    %map edges
    imR = goldenSiftResults{ind, 8};
    imC = goldenSiftResults{ind, 9};
    H = Hs{ind};
    invH = inv(H);
    border = 5;
    
    pa = zeros(2,4);
    pa(:,1) = [-border; -border];
    pa(:,2) = [-border; imR+border];
    pa(:,3) = [imC+border; imR+border];
    pa(:,4) = [imC+border; -border];

    imshow(rawImage)
    hold on
    
    for i=1:numel(pa)/2
        paInUse = pa(:,i);
        pb = T*paInUse + C;
        xSample = pb(1);
        ySample = pb(2);
        plot(round(xSample),round(ySample),'ok');
    end
    
    pa = [pa; ones(1, numel(pa)/2)];
    for i=1:numel(pa)/3
        paInUse = pa(:,i);
        pb = invH*paInUse;
        pb = pb/pb(3);
        xSample = pb(1);
        ySample = pb(2);
        plot(round(xSample),round(ySample),'*k');
    end
end

outputStruct.Country = { goldenSiftResults{ind, 1} };
outputStruct.ImgLoc = { goldenSiftResults{ind, 2} };
outputStruct.Features = { Dsamp };
outputStruct.matchSum = { matchSum };

end