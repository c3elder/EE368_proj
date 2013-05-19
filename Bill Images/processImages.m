fid = fopen('images.csv','r');
C = textscan(fid, repmat('%s',1,4), 'delimiter',',', 'CollectOutput',true);
C = C{1};
fclose(fid);

folderSlash = '\';
if isunix
    folderSlash = '/';
end

[rows, cols] = size(C);

goldenSiftResults=cell(rows-1,7);

% USE DEFAULT THRESHOLD OF
defaultThresh = 17;

for i=1:rows-1
    country = C{i+1,1}
    location = C{i+1,2};
    value = str2double(C{i+1,3});
    thresh = str2double(C{i+1,4});
    if thresh==-1
        thresh = defaultThresh;
    end
    
    [f, d] = vl_sift(single(rgb2gray(imread([country, folderSlash, location]))), 'PeakThresh', thresh);
    kdTree = vl_kdtreebuild(single(d));
    goldenSiftResults(i,:) = {country, location, value, thresh, f, single(d), kdTree};
    %memory
end

save('goldenSiftResults.mat', 'goldenSiftResults');