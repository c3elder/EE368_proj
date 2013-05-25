folderSlash = '\';
if isunix
    folderSlash = '/';
end

fid = fopen(['Sample Images' folderSlash 'tester.csv'],'r');
C = textscan(fid, repmat('%s',1,8), 'delimiter',',', 'CollectOutput',true);
C = C{1};
fclose(fid);

[rows, cols] = size(C);
results = zeros(rows-1, 3);

%test every image against the expected results
algorithmHandle = @matchImagesNoSeg;

outputStructArray = {};

for i=2:rows
    inputImagePath = ['Sample Images' folderSlash C{i, 1} folderSlash C{i, 2}];
    outputStruct = algorithmHandle(inputImagePath);
    
    countryMatch = {C{i,3}, C{i,4}, C{i,5}};
    locationMatch = {C{i,6}, C{i,7}, C{i,8}};
    resultsToCompare = cellfun(@isempty, countryMatch);
    
    %make sure that each of the above results is matched by a returned
    %value
    countryResults = outputStruct.Country;
    imgLocResults = outputStruct.ImgLoc;
    for j=1:numel(countryResults)
        tmp = strcmp(countryResults{j}, countryMatch) & strcmp(imgLocResults{j}, locationMatch) ;
        resultsToCompare = tmp | resultsToCompare; 
    end
    
    fprintf('Test %d had result %d with %d orig matches and %d processed matches \n', i-1, outputStruct.matchSumOrig{1}, outputStruct.matchSum{1});
    outputStructArray{end+1} = outputStruct;
    close all;
end