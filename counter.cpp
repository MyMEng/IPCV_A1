#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <cmath>

#define NIMAGES 1
#define PI 3.14159265

#define IMGTHRESHOLD 85
#define HOUGHX 200 //dimension of Hough Space in x direction
#define HOUGHY 200 //dimension of Hough Space in y direction
#define RMAX 100 //maximal radius of circle
#define DELTATHETA 0.26179939 //15 degrees in radians

void sobel(const cv::Mat& image, int number, cv::Mat& xDeriv, cv::Mat& yDeriv, cv::Mat& grad, cv::Mat& arc)
{
	// Compute and display image containing the derivative in the x direction af/ax
	std::ostringstream windowName;
	double minVal, maxVal;

	cv::Mat xNorm;
	windowName << "Coins " << (number + 1) << ": X derivative";
	cv::Sobel(image, xDeriv, CV_64F, 1, 0);

	cv::normalize(xDeriv, xNorm, 0, 255, cv::NORM_MINMAX);
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::Mat temp8Bit;
	xNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(),temp8Bit);

	// Compute and display image containing the derivative in the x direction af/ax
	cv::Mat yNorm;

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
	cv::Mat xPow, yPow;
	cv::pow(xDeriv, 2, xPow);
	cv::pow(yDeriv, 2, yPow);
	cv::Mat temp = xPow + yPow;

	cv::Mat tempFloat;
	temp.convertTo(tempFloat, CV_64F);
	cv::Mat gradNorm;
	cv::sqrt(tempFloat, grad);

	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": gradient magnitude";
	
	cv::normalize(grad, gradNorm, 0, 255, cv::NORM_MINMAX);
	gradNorm.convertTo(temp8Bit, CV_8U);

	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), temp8Bit);

	// Arc tan
	// Change window name
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (number + 1) << ": arc Tangent";
	cv::Mat divided, arcNorm;
	arc = ( cv::Mat(xNorm.rows, xNorm.cols, CV_64F) ).clone() ;
	// arc.rows = xNorm.rows ;
	// arc.cols = xNorm.cols ;
	// arc.type = CV_64F ;

	cv::divide(yDeriv, xDeriv, divided);
	for(int i = 0; i < divided.rows; i++)
	{
		for(int j =0; j < divided.cols; j++)
		{	
			arc.at<double>(i, j) = (double)atan(divided.at<double>(i, j)) ;//* 180 / PI;
		}
	} 

	// Display minimum and maximum value from matrix
	// cv::minMaxLoc(arc, &minVal, &maxVal) ;
	// std::cout << "Minval: " << minVal << " Maxval: " << maxVal << std::endl ;

	cv::normalize(arc, arcNorm, 0, 255, cv::NORM_MINMAX);

	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	arcNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(), temp8Bit);

	// Phase Calculations
	// Change window name
	// windowName.str("");
	// windowName.clear();
	// windowName << "Coins " << (number + 1) << ": opencv phase";

	// cv::phase(xDeriv, yDeriv, arc);
	// cv::normalize(arc, arcNorm, 0, 255, cv::NORM_MINMAX);

	// cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	// arcNorm.convertTo(temp8Bit, CV_8U);
	// cv::imshow(windowName.str().c_str(), temp8Bit);

}

cv::Mat trshld( const int imageID, const cv::Mat& xDeriv, const cv::Mat& yDeriv, const cv::Mat& grad, const cv::Mat& arc )
{
	std::ostringstream windowName;

	//threshold the gradient image after normalization
	cv::Mat gradNorm ;
	cv::normalize(grad, gradNorm, 0, 255, cv::NORM_MINMAX);
	for(int i = 0; i < gradNorm.rows; ++i)
	{
		for (int j = 0; j < gradNorm.cols; ++j)
		{
			//std::cout << "i: " << i << " j: " << j << std::endl ;
			double tr = gradNorm.at<double>(i, j);
			if (tr > IMGTHRESHOLD)
			{
				gradNorm.at<double>(i, j) = 255 ;
			} else
			{
				gradNorm.at<double>(i, j) = 0 ;
			}
		}
	}
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (imageID + 1) << ": gradient thresholded";
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), gradNorm) ;

	return gradNorm ;
}

void hough( const int imageID, cv::Mat& grad, const cv::Mat& arc)
{
	std::ostringstream windowName;

	// threshold the gradient image after normalization
	cv::Mat gradNorm(grad.rows, grad.cols, CV_8U) ;
	// cv::normalize(grad, gradNorm, 0, 255, cv::NORM_MINMAX);

	int counter = 0 ;
	int cprim = 0;

	for(int i = 0; i < grad.rows; ++i)
	{
		for (int j = 0; j < grad.cols; ++j)
		{
			//std::cout << "i: " << i << " j: " << j << std::endl ;
			if (grad.at<double>(i, j) == 255)
			{
				for (int r = 0; r < RMAX; ++r)
				{
					double x1 = i+r*cos(arc.at<double>(i,j));
					double x2 = i-r*cos(arc.at<double>(i,j));
					double y1 = j+r*sin(arc.at<double>(i,j));
					double y2 = j-r*sin(arc.at<double>(i,j));

					if (x1 == x2 && y1 == y2)
					{
						counter++ ;
						std::cout << "Eureka!" << x1 << "  " << y1 << std::endl ;
						// fist we define the properties that the circle will have.
					    cv::Scalar redColour(255, 0, 0);
					    int radius = r;

					    // providing a negative number will create a filled circle
					    int thickness = 10;

					    // 8 connected line
					    // ( also there is a 4 connected line and CVAA which is an anti aliased line )
					    int linetype = 8; 

					    // here is where we define the center of the circle
        				cv::Point center( x1, y1 );
        				cv::circle ( gradNorm , center , radius , 50 , thickness , linetype );

						//remember about transforming form i, j coordinates to hough coordinate
						//increase hough space for x1, y1, r
					}
					cprim++ ;
				}
			}
		}
	}
	std::cout << "Counter: " << counter << " Cprim: " << cprim << std::endl ;

	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (imageID + 1) << ": circles";
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), gradNorm) ;
}

int main( int argc, char ** argv )
{
	// Coins
	cv::Mat coins[3];
	coins[0] = cv::imread("coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
	coins[1] = cv::imread("coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
	coins[2] = cv::imread("coins3.png", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat xD, yD, grad, angle;

	std::ostringstream windowName;

	for(int i = 0; i < NIMAGES; ++i)
	{
		windowName.str("");
		windowName.clear();
		windowName << "Coins " << (i+1);

		cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName.str().c_str(), coins[i]);
		sobel(coins[i], i, xD, yD, grad, angle);

		//edge extraction using Hough Circle Detection
		cv::Mat gradTr = trshld(i, xD, yD, grad, angle) ;
		hough(i, gradTr, angle) ;
	}

	cv::waitKey();

	return 0;
}

