US_one_front = rgb2gray(imread('Bill Images/US/one_dollar_font.jpg'));
[US_one_front_feat,US_one_front_descrip] = vl_sift(single(US_one_front));

US_one_back = rgb2gray(imread('Bill Images/US/one_dollar_back.jpg'));
[US_one_back_feat,US_one_back_descrip] = vl_sift(single(US_one_back));

US_five_front_1999 = rgb2gray(imread('Bill Images/US/five_front_1999.jpg'));
[US_five_front_1999_feat,US_five_front_1999_descrip] = vl_sift(single(US_five_front_1999));

US_five_front_2006 = rgb2gray(imread('Bill Images/US/five_front_2006.jpg'));
[US_five_front_2006_feat,US_five_front_2006_descrip] = vl_sift(single(US_five_front_2006));

US_five_back_2006 = rgb2gray(imread('Bill Images/US/five_back_2006.jpg'));
[US_five_back_2006_feat,US_five_back_2006_descrip] = vl_sift(single(US_five_back_2006));

save('Bill Images/SIFT_results/US_bills.mat')
