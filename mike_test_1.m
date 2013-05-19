%% ubc_match search through the entire feature vector space, wayyyy to slow
load('Bill Images/SIFT_results/US_bills.mat');

samplef1 = rgb2gray(imread('Sample Images/US/one_front.jpg'));
[Fsamp1,Dsamp1] = vl_sift(single(samplef1));

%samplef5 = rgb2gray(imread('Sample Images/US/five_front_2006.jpg'));
%[Fsamp5,Dsamp5] = vl_sift(single(samplef5));

[matches1, scores1] = vl_ubcmatch(US_one_front_descrip, Dsamp1);
[matches2, scores2] = vl_ubcmatch(US_one_back_descrip, Dsamp1);
[matches3, scores3] = vl_ubcmatch(US_five_front_1999_descrip, Dsamp1);
[matches4, scores4] = vl_ubcmatch(US_five_front_2006_descrip, Dsamp1);
[matches5, scores5] = vl_ubcmatch(US_five_back_2006_descrip, Dsamp1);

%% use Flann or kdtree for searching
% http://www.vlfeat.org/overview/kdtree.html
golden = rgb2gray(imread('Bill Images/US/one_dollar_front.jpg'));
[Fgolden,Dgolden] = vl_sift(single(golden), 'PeakThresh', 0.05);
front = single(Dgolden);

load('Bill Images/SIFT_results/US_bills.mat');
front = single(US_one_front_descrip);
kdtree = vl_kdtreebuild(front);

samplef1 = rgb2gray(imread('Sample Images/US/one_front.jpg'));
[Fsamp1,Dsamp1] = vl_sift(single(samplef1));
sample = single(Dsamp1);

%% 
featureMatch = zeros(length(sample), 2);
for i=1:length(sample)
    [index, distance] = vl_kdtreequery(kdtree, front, sample(:,i));
    featureMatch(i, :) = [index, distance];
end


