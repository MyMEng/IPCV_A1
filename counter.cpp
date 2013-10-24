#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>

int main( int argc, char ** argv )
{
	cv::Mat coins1 = cv::imread("images/coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat coins2 = cv::imread("images/coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat coins3 = cv::imread("images/coins3.png", CV_LOAD_IMAGE_GRAYSCALE);


	cv::namedWindow("Coins 1", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Coins 2", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Coins 3", CV_WINDOW_AUTOSIZE);

	cv::imshow("Coins 1", coins1);
	cv::imshow("Coins 2", coins2);
	cv::imshow("Coins 3", coins3);

	cv::waitKey();

	return 0;
}