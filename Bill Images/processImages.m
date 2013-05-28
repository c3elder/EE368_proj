fid = fopen('images.csv','r');
C = textscan(fid, repmat('%s',1,4), 'delimiter',',', 'CollectOutput',true);
C = C{1};
fclose(fid);

folderSlash = '\';
if isunix
    folderSlash = '/';
end

[rows, cols] = size(C);

goldenSiftResults=cell(rows-1,9);

% USE DEFAULT THRESHOLD OF
thresh = 0.5;

maxFeatures = 500;

for i=1:rows-1
    country = C{i+1,1}
    location = C{i+1,2};
    value = str2double(C{i+1,3});
    unitValue = str2double(C{i+1,4});
    
    imInUse = imread([country, folderSlash, location]);
    [imR, imC, N] = size(imInUse);
    %imInUse = rgb2ycbcr(imInUse);
    %[f, d] = vl_sift(single(imInUse(:,:,1)), 'PeakThresh', thresh);
    [f, d] = vl_sift(single(rgb2gray(imInUse)), 'PeakThresh', thresh);
    
    %eliminate features with very small scale
    [sortedList sortedInd ] = sort(f(1,:), 'descend');
    smallestFeatNum = min(size(f,2), maxFeatures);
    if smallestFeatNum < maxFeatures
       fprintf('not enough features\n'); 
    end
    f = f(:, sortedInd(1:smallestFeatNum));
    d = d(:, sortedInd(1:smallestFeatNum));
    
    kdTree = vl_kdtreebuild(single(d));
    goldenSiftResults(i,:) = {country, location, value, thresh, f, single(d), unitValue, imR, imC};
end

save('goldenSiftResults.mat', 'goldenSiftResults');