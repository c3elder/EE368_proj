function outputStruct = matchImagesNoSeg(inputImagePath)

load('Bill Images/goldenSiftResults.mat');

rawImage = imread(inputImagePath);

% Christian's processing goes here

[M N C] = size(rawImage);
if C==3
    sample = rgb2gray(rawImage);
else
    sample = rawImage;
end

outputStruct = struct();


[Fsamp,Dsamp] = vl_sift(single(sample), 'PeakThresh', 5);
Dsamp = single(Dsamp);

[goldenRows, goldenCols] = size(goldenSiftResults);
matchSum = zeros(goldenRows,1);

for j = 1:goldenRows;
    [DsampRows, DsampCols] = size(Dsamp);
    featureMatch = zeros(DsampCols, 2);
    for i=1:DsampCols
        % use Flann or kdtree for searching
        % http://www.vlfeat.org/overview/kdtree.html
        
        [index, distance] = vl_kdtreequery(goldenSiftResults{j,7}, goldenSiftResults{j,6}, Dsamp(:,i));
        featureMatch(i, :) = [index, distance];
    end
    [featureMatchSorted ind] = sort(featureMatch(:, 2));
    matchSum(j) = sum(featureMatchSorted(1:20));
end

[val ind]= min(matchSum);

outputStruct.Country = { goldenSiftResults{ind, 1} };
outputStruct.ImgLoc = { goldenSiftResults{ind, 2} };
outputStruct.Features = { Dsamp };
outputStruct.matchSum = { matchSum };