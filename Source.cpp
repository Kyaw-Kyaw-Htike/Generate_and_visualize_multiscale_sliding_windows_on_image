// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "opencv2/opencv.hpp"
#include <iostream>
#include <tuple>
#include <functional>

/*
Given an image, generate multiscale sliding windows. Also optionally extract 
feature vector for each sliding window. If does not want feature extraction,
simply pass nullptr to the argument feature_extractor. Otherwise, the feature extraction
can be anything as long as it follows the signature expected by feature_extractor.
*/

std::vector<double> extract_raw_pixels(cv::Mat img)
{
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img, CV_64FC1, 1.0 / 255.0);
	std::vector<double> vout(img.begin<double>(), img.end<double>());
	return vout;
}

//std::vector<double> extract_color_histogram(cv::Mat img)
//{
//	// etc.
//}
//

std::tuple<std::vector<cv::Rect>, std::vector<std::vector<double>>> multiscale_slidewins_image(const cv::Mat & img, int winsize_rows, int winsize_cols, double scaleratio, int max_nscales, int stride, std::function<std::vector<double>(cv::Mat)> feature_extractor )
{		
	// no. of scales that image sliding window must process
	int num_scales;
	// vector of scales computed for sliding window; scales.size()==num_scales
	std::vector<double> scales;
	// total no. of sliding windows for the image (across all the scales).
	unsigned int nslidewins_total;
	// vector of sliding window rectangles. dr.size()==nslidewins_total
	std::vector<cv::Rect> dr;
	// vector of feature vectors extracted
	std::vector<std::vector<double>> feats;
	// for each sliding window rectangle, which scale did it come from;
	// stores the index to std::vector<double>scales
	std::vector<unsigned int> idx2scale4dr;
	int winsize[] = {winsize_rows, winsize_cols};

	bool extract_feat = false;
	if (feature_extractor)
		extract_feat = true;

	int nrows_img = img.rows; // for use locally in this method
	int ncols_img = img.cols; // for use locally in this method

	// compute analytically how many scales there are for sliding window.
	// this formula gives the same answer as would be computed in a loop.
	num_scales = std::min(std::floor(std::log(static_cast<double>(nrows_img) / winsize[0]) / std::log(scaleratio)),
		std::floor(std::log(static_cast<double>(ncols_img) / winsize[1]) / std::log(scaleratio))) + 1;

	// preallocate for efficiency
	scales.resize(num_scales);

	// find a tight upper bound on total no. of sliding windows needed
	double stride_scale, nsw_rows, nsw_cols;
	size_t nslidewins_total_ub = 0;
	for (size_t s = 0; s < num_scales; s++)
	{
		stride_scale = stride*std::pow(scaleratio, s);
		nsw_rows = std::floor(nrows_img / stride_scale) - std::floor(winsize[0] / stride) + 1;
		nsw_cols = std::floor(ncols_img / stride_scale) - std::floor(winsize[1] / stride) + 1;
		// Without the increment below, I get exact computation of number of sliding
		// windows, but just in case (to upper bound it)
		++nsw_rows; ++nsw_cols;
		nslidewins_total_ub += (nsw_rows * nsw_cols);
	}

	// preallocate/reserve for speed		
	dr.reserve(nslidewins_total_ub);
	idx2scale4dr.reserve(nslidewins_total_ub);
	if (extract_feat)
	{
		feats.reserve(nslidewins_total_ub);
	}

	// the resized image
	cv::Mat img_cur;
	// reset counter for total number of sliding windows across all scales
	nslidewins_total = 0;

	for (size_t s = 0; s < num_scales; s++)
	{
		// compute how much I need to scale the original image for this current scale s
		scales[s] = std::pow(scaleratio, s);
		// get the resized version of the original image with the computed scale
		cv::resize(img, img_cur, cv::Size(), 1.0 / scales[s], 1.0 / scales[s], cv::INTER_LINEAR);

		// run sliding window in the channel image space 
		for (size_t i = 0; i < img_cur.rows - winsize[0] + 1; i += stride)
		{
			for (size_t j = 0; j < img_cur.cols - winsize[1] + 1; j += stride)
			{
				dr.push_back(cv::Rect(
					std::round(j*scales[s]),
					std::round(i*scales[s]),
					std::round(winsize[1] * scales[s]),
					std::round(winsize[0] * scales[s]))
				);

				// stores which scale of the original image this dr comes from
				idx2scale4dr.push_back(s);

				// save the extracted features
				if (extract_feat)
				{
					feats.push_back(feature_extractor(img_cur(cv::Rect(j, i, winsize[1], winsize[0]))));
				}

				++nslidewins_total;

			} // end j
		} //end i

	} //end s

	return std::make_tuple(dr, feats);

} 

int main()
{
	cv::Mat img = cv::imread("D:/Research/Datasets/CUHK_Square/frames_train/Culture_Square_00151.png");
	auto dr_feats = multiscale_slidewins_image(img, 90, 90, 2, 2, 32, extract_raw_pixels);
	std::vector<cv::Rect> dr = std::get<0>(dr_feats);
	std::vector<std::vector<double>> feats = std::get<1>(dr_feats);
	printf("Number of multiscale sliding windows = %d\n", dr.size());
	for (size_t i = 0; i < dr.size(); i++)
	{
		cv::Mat img2 = img.clone();
		cv::rectangle(img2, cv::Rect(308, 256, 44, 74), cv::Scalar(255, 0, 0), 3);
		cv::rectangle(img2, dr[i], cv::Scalar(0, 0, 255), 3);
		cv::imshow("win", img2);
		cv::waitKey(1);
	}
	return 0;
}