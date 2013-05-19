function outputStruct = matchImages(inputImagePath)

% load('Bill Images/goldenSiftResults.mat'); NOTE: Loaded below

rawImage = im2double(imread(inputImagePath));

% **********************BEGIN SEGMENTATION*********************************
fprintf('Segmenting...\n')
% Downsample
downsize = 1/4;
img_ds = imresize(rawImage,downsize);

% figure(1);imshow(img)
% clear img
% figure(2);
% subplot(2,2,1);imshow(img_ds)

% Convert to HSV
img_ds_hsv = rgb2hsv(img_ds);
% subplot(2,2,2);imshow(img_ds_hsv(:,:,1))
% subplot(2,2,3);imshow(img_ds_hsv(:,:,2))
% subplot(2,2,4);imshow(img_ds_hsv(:,:,3))

% Threshold
thresh_global = graythresh(img_ds);
thresh_h = graythresh(img_ds_hsv(:,:,1));
thresh_s = graythresh(img_ds_hsv(:,:,2));
thresh_v = graythresh(img_ds_hsv(:,:,3));

% Threshold
img_ds_bw = im2bw(img_ds,thresh_global);
img_h_bw = im2bw(img_ds_hsv(:,:,1),thresh_h);
img_s_bw = im2bw(img_ds_hsv(:,:,2),thresh_s);
img_v_bw = im2bw(img_ds_hsv(:,:,3),thresh_v);

% figure(3);
% subplot(2,2,1);imshow(img_ds_bw)
% subplot(2,2,2);imshow(img_h_bw)
% subplot(2,2,3);imshow(img_s_bw)
% subplot(2,2,4);imshow(img_v_bw)

% Apply LoG Operator
img_ds_log = edge(img_ds_bw,'log');
img_h_log = edge(img_h_bw,'log');
img_s_log = edge(img_s_bw,'log');
img_v_log = edge(img_v_bw,'log');

% figure(4);
% subplot(2,2,1);imshow(img_ds_log)
% subplot(2,2,2);imshow(img_h_log)
% subplot(2,2,3);imshow(img_s_log)
% subplot(2,2,4);imshow(img_v_log)

% Dilate
box = ones(7,7);
img_ds_dil = imdilate(img_ds_log,box);
img_h_dil = imdilate(img_h_log,box);
img_s_dil = imdilate(img_s_log,box);
img_v_dil = imdilate(img_v_log,box);

% figure(5);
% subplot(2,2,1);imshow(img_ds_dil)
% subplot(2,2,2);imshow(img_h_dil)
% subplot(2,2,3);imshow(img_s_dil)
% subplot(2,2,4);imshow(img_v_dil)

% Fill Holes
img_ds_fill = imfill(img_ds_dil,'holes');
img_h_fill = imfill(img_h_dil,'holes');
img_s_fill = imfill(img_s_dil,'holes');
img_v_fill = imfill(img_v_dil,'holes');

% figure(6)
% subplot(2,2,1);imshow(img_ds_fill)
% subplot(2,2,2);imshow(img_h_fill)
% subplot(2,2,3);imshow(img_s_fill)
% subplot(2,2,4);imshow(img_v_fill)

% Label Regions
img_v_label = bwlabel(img_v_fill,8);
imgProps = regionprops(img_v_label, 'Solidity', 'Area', 'MajorAxisLength', 'MinorAxisLength','Orientation');

% Keep blobs with 'bill-like' eccentricity
img_v_final = img_v_fill;
for nRegion = 1:length(imgProps)
    idx = find(img_v_label == nRegion);
    maj_len = imgProps(nRegion).MajorAxisLength;
    min_len = imgProps(nRegion).MinorAxisLength;
    eccentricity = maj_len/min_len;
    if ((eccentricity < 2) || ...
        (eccentricity > 4) || ...
        (min_len < 50) || ...
        (imgProps(nRegion).Solidity < 0.8) || ...
        (imgProps(nRegion).Area < 600))
    
        %(min_len < 50) || ...
        %(maj_len > 600) || ...
        img_v_final(idx) = 0;
    else
        [maj_len, min_len, imgProps(nRegion).Orientation];
        
    end
end % nRegion

img_h_label = bwlabel(img_h_fill,8);
imgProps = regionprops(img_h_label, 'Solidity', 'Area', 'MajorAxisLength', 'MinorAxisLength','Orientation');

% Keep blobs with 'bill-like' eccentricity
img_h_final = img_h_fill;
for nRegion = 1:length(imgProps)
    idx = find(img_h_label == nRegion);
    maj_len = imgProps(nRegion).MajorAxisLength;
    min_len = imgProps(nRegion).MinorAxisLength;
    eccentricity = maj_len/min_len;
    if ((eccentricity < 1.5) || ...
        (eccentricity > 4) || ...
        (min_len < 50) || ...
        (imgProps(nRegion).Solidity < 0.8) || ...
        (imgProps(nRegion).Area < 600))
        %(min_len < 50) || ...
        %(maj_len > 600) || ...
        img_h_final(idx) = 0;
    else
        [maj_len, min_len, imgProps(nRegion).Orientation];
        
    end
end % nRegion

% figure(7); imshow(img_v_final)
% figure(8); imshow(rgb2gray(img_ds).*img_v_final)
% figure(9); imshow(img_h_final)
% figure(10);imshow(rgb2gray(img_ds).*img_h_final)

% Combine Filters
img_hv_all = ((img_h_final + img_v_final) > 0);
figure(11);imshow(img_hv_all)

% Separate the Bills
img_hv_label = bwlabel(img_hv_all,8);
imgProps = regionprops(img_hv_label, 'Orientation','ConvexImage','BoundingBox');

% Write Each Bill Mask to a separate image
img_hv_final = zeros([size(img_hv_all),length(imgProps)]);%repmat(img_hv_all,[1,1,length(imgProps)]);
for nRegion = 1:length(imgProps)
    idx = find(img_hv_label == nRegion);
    convHull = imgProps(nRegion).ConvexImage;
    box = imgProps(nRegion).BoundingBox;
%     disp([box(3), box(4), size(convHull)])
    
    img_hv_final(box(2):box(2)+box(4)-1,box(1):box(1)+box(3)-1,nRegion) = convHull;
    figure(11+nRegion);imshow(img_hv_final(:,:,nRegion))
%    img_hv_final(:,:,nRegion) = imgProps(nRegion).ConvexImage;
    % Black out blob nRegion in all but the nRegion-th image
%     for i = 1:length(imgProps)
%         if (i ~= nRegion)
%             img_tmp = img_hv_final(:,:,i);
%             img_tmp(idx) = 0;
%             img_hv_final(:,:,i) = img_tmp;
%         end
%     end
end

% ****************END OF SEGMENTATION**************************************
fprintf('End Segmentation, Start SIFTing...\n')

% **********************START SIFTING**************************************
% sample = rgb2gray(rawImage);

clearvars -except rawImage img_hv_final downsize
load('Bill Images/goldenSiftResults.mat');
num_bills = size(img_hv_final,3);
outputStruct = struct();  
outputStruct.Country = cell(1,num_bills);
outputStruct.ImgLoc = cell(1,num_bills);
outputStruct.Features = cell(1,num_bills);

% SIFT each bill individually
for bill = 1:size(img_hv_final,3)
    disp(sprintf('SIFTING Bill# %d',bill))
    sample  = rgb2gray(rawImage.* ...
              repmat(imresize(img_hv_final(:,:,bill),1/downsize,'nearest'),[1,1,3]));
    figure(bill+11);imshow(sample);title(sprintf('Bill#: %d',bill))
    
    [Fsamp,Dsamp] = vl_sift(single(sample), 'PeakThresh', 0.05);
    Dsamp = single(Dsamp);
%     disp(Dsamp)
    
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
    
    outputStruct.Country{bill} = goldenSiftResults{ind, 1};
    outputStruct.ImgLoc{bill} = goldenSiftResults{ind, 2};
    outputStruct.Features{bill} = Dsamp;
    
end