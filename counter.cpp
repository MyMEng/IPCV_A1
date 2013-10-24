#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>

#define NIMAGES 1

void sobel(const cv::Mat& image, int number)
{
	// Compute and display image containing the derivative in the x direction af/ax
	std::ostringstream windowName;
	cv::Mat xDeriv;

	windowName << "Coins " << (number + 1) << ": X derivative";

	cv::Sobel(image, xDeriv, CV_8U, 1, 0);

	//double minVal, maxVal;
	//cv::minMaxLoc(xDeriv, &minVal, &maxVal);
	cv::normalize(xDeriv, xDeriv, 0, 255, cv::NORM_MINMAX);


	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), xDeriv);

	// Compute and display image containing the derivative in the x direction af/ax
	cv::Mat yDeriv;

	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": Y derivative";
	
	cv::Sobel(image, yDeriv, CV_8U, 0, 1);
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), yDeriv);

	// Image containing magnitude of the gradient f(x,y)

	//cv::Mat temp = xDeriv.mul(xDeriv) + yDeriv.mul(yDeriv);
	cv::Mat xPow, yPow;
	cv::pow(xDeriv, 2, xPow);
	cv::pow(yDeriv, 2, yPow);
	cv::Mat temp = xPow + yPow;

	cv::Mat tempFloat;
	temp.convertTo(tempFloat, CV_64F);
	cv::Mat grad;
	cv::sqrt(tempFloat, grad);

	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": gradient magnitude";
	
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), grad);


}

int main( int argc, char ** argv )
{
	// Coins
	cv::Mat coins[3];
	coins[0] = cv::imread("coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
	coins[1] = cv::imread("coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
	coins[2] = cv::imread("coins3.png", CV_LOAD_IMAGE_GRAYSCALE);

	std::ostringstream windowName;

	for(int i = 0; i < NIMAGES; ++i)
	{
		windowName.str("");
		windowName.clear();
		windowName << "Coins " << (i+1);

		cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName.str().c_str(), coins[i]);
		sobel(coins[i], i);
	}

	cv::waitKey();

	return 0;
}

