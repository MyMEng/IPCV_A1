#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <cmath>

#define NIMAGES 1
#define PI 3.14159265

void sobel(const cv::Mat& image, int number)
{
	// Compute and display image containing the derivative in the x direction af/ax
	std::ostringstream windowName;
	cv::Mat xDeriv, xNorm;

	windowName << "Coins " << (number + 1) << ": X derivative";

	cv::Sobel(image, xDeriv, CV_64F, 1, 0);

	double minVal, maxVal;
	cv::normalize(xDeriv, xNorm, 0, 255, cv::NORM_MINMAX);
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::Mat temp8Bit;
	xNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(),temp8Bit);

	// Compute and display image containing the derivative in the x direction af/ax
	cv::Mat yDeriv, yNorm;

	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": Y derivative";
	
	cv::Sobel(image, yDeriv, CV_64F, 0, 1);
	cv::normalize(yDeriv, yNorm, 0, 255, cv::NORM_MINMAX);

	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	yNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(), temp8Bit);

	// Image containing magnitude of the gradient f(x,y)

	//cv::Mat temp = xDeriv.mul(xDeriv) + yDeriv.mul(yDeriv);
	cv::Mat xPow, yPow;
	cv::pow(xDeriv, 2, xPow);
	cv::pow(yDeriv, 2, yPow);
	cv::Mat temp = xPow + yPow;

	cv::Mat tempFloat;
	temp.convertTo(tempFloat, CV_64F);
	cv::Mat grad, gradNorm;
	cv::sqrt(tempFloat, grad);

	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": gradient magnitude";
	
	cv::normalize(grad, gradNorm, 0, 255, cv::NORM_MINMAX);

	for(int i = 0; i < gradNorm.rows; i++)
	{
		for(int j =0; j < gradNorm.cols; j++)
		{
			double point = gradNorm.at<double>(i, j);
			gradNorm.at<double>(i, j) = point - minVal;
		}
	}


	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	gradNorm.convertTo(temp8Bit, CV_8U);


	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), temp8Bit);

	// Arc tan
	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": arctan";
	cv::Mat divided, arc, arcNorm;
	cv::divide(yDeriv, xDeriv, divided);


	arc = divided.clone();

	for(int i = 0; i < divided.rows; i++)
	{
		for(int j =0; j < divided.cols; j++)
		{
			double point = divided.at<double>(i, j);
			arc.at<double>(i, j) = (double)atan(point) * 180 / PI;
		}
	}

	cv::normalize(arc, arcNorm, 0, 255, cv::NORM_MINMAX);


	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	arcNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(), temp8Bit);
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

