#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <cmath>

#define NIMAGES 3
#define PI 3.14159265

#define IMGTHRESHOLD 65
#define HOUGHX 110//441//110 //147 //dimension of Hough Space in cols
#define HOUGHY 85//341//85 //113 //dimension of Hough Space in rows
#define RMIN 30
#define RMAX 55 //maximal radius of circle
#define HOUGHTHRESHOLD 95//24 //95

#define MEDIANFILTERWIDTH 3
#define MEDIANFILTERHEIGHT 6

#define INTERVAL 17 //confidence interval for circles

using namespace cv;
using namespace std;


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
			arc.at<double>(i, j) = (double)atan2(yDeriv.at<double>(i,j), xDeriv.at<double>(i,j)) ;//* 180 / PI;
		}
	} 

	cv::normalize(arc, arcNorm, 0, 255, cv::NORM_MINMAX);

	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	arcNorm.convertTo(temp8Bit, CV_8U);
	cv::imshow(windowName.str().c_str(), temp8Bit);

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

void hough( const int imageID, cv::Mat& grad, const cv::Mat& arc, cv::Mat& img)
{
	std::cout << "Achieved this stage 0" << std::endl ;
	std::ostringstream windowName;
	cv::vector<cv::Vec3d> circles ; //x,y,r
	std::vector<std::vector<std::vector<int> > > houghSpace (HOUGHY, std::vector<std::vector<int> > (HOUGHX, std::vector<int>(RMAX-RMIN, 0) ) ) ;
	cv::Mat flatHoughSpace = cv::Mat(grad.rows, grad.cols, CV_64F, cv::Scalar::all(0));

	// threshold the gradient image after normalization
	cv::Mat gradNorm(grad.rows, grad.cols, CV_64F) ;

	std::vector <double> xes ;
	std::vector <double> yes ;

	//remember about transforming form i, j coordinates to hough coordinate
	int rowMin = 0 ; // 0 - RMAX ;
	int colMin = 0 ; // 0 - RMAX ;
	int rowMax = grad.rows -1 + 2*RMAX ; // grad.rows + RMAX ;
	int colMax = grad.cols -1 + 2*RMAX ; // grad.cols + RMAX ;

	for(int i = 0; i < grad.rows; ++i)
	{
		for (int j = 0; j < grad.cols; ++j)
		{
			if (grad.at<double>(i, j) == 255)
			{
				for (int r = RMIN; r < RMAX; ++r)
				{
					//shifted by RMAX to make scaling easier task
					double x1 = j+r*cos(arc.at<double>(i,j)) ;
					double x2 = j-r*cos(arc.at<double>(i,j)) ;
					double y1 = i+r*sin(arc.at<double>(i,j)) ;
					double y2 = i-r*sin(arc.at<double>(i,j)) ;

					xes.push_back(x1) ;
					xes.push_back(x2) ;
					yes.push_back(y1) ;
					yes.push_back(y2) ;

					x1 += RMAX ;
					x2 += RMAX ;
					y1 += RMAX ;
					y2 += RMAX ;

					int sx1 = round( ( ( x1 * (HOUGHX-1) / (colMax - colMin) ) ) ) ;
					int sx2 = round( ( ( x2 * (HOUGHX-1) / (colMax - colMin) ) ) ) ;
					int sy1 = round( ( ( y1 * (HOUGHY-1) / (rowMax - rowMin) ) ) ) ;
					int sy2 = round( ( ( y2 * (HOUGHY-1) / (rowMax - rowMin) ) ) ) ;

					if ( sx1 > HOUGHX || sx2 > HOUGHX || sy1 > HOUGHY || sy2 > HOUGHY )
					std::cout << sx1 << " " << sx2 << " " << sy1 << " " << sy2 << std::endl;

					//increase hough space for x1, y1, r
					houghSpace[sy1][sx1][r-RMIN] += 1 ;
					houghSpace[sy2][sx1][r-RMIN] += 1 ;
					houghSpace[sy1][sx2][r-RMIN] += 1 ;
					houghSpace[sy2][sx2][r-RMIN] += 1 ;

					flatHoughSpace.at<double>( round ( ( y1 * (flatHoughSpace.rows-1) / (rowMax - rowMin) ) ), round ( ( x1 * (flatHoughSpace.cols-1) / (colMax - colMin) ) ) ) += 1 ;
					flatHoughSpace.at<double>( round ( ( y1 * (flatHoughSpace.rows-1) / (rowMax - rowMin) ) ), round ( ( x2 * (flatHoughSpace.cols-1) / (colMax - colMin) ) ) ) += 1 ;
					flatHoughSpace.at<double>( round ( ( y2 * (flatHoughSpace.rows-1) / (rowMax - rowMin) ) ), round ( ( x1 * (flatHoughSpace.cols-1) / (colMax - colMin) ) ) ) += 1 ;
					flatHoughSpace.at<double>( round ( ( y2 * (flatHoughSpace.rows-1) / (rowMax - rowMin) ) ), round ( ( x2 * (flatHoughSpace.cols-1) / (colMax - colMin) ) ) ) += 1 ;

				}
			}
		}
	}
	std::cout << "Achieved this stage2" << std::endl ;

	//take logs
	for (int i = 0; i < flatHoughSpace.rows; ++i)
	{
		for (int j = 0; j < flatHoughSpace.cols; ++j)
		{
			if (flatHoughSpace.at<double>(i,j) != 0)
			{
				flatHoughSpace.at<double>(i,j) = log( flatHoughSpace.at<double>(i,j) ) ;
			}
		}
	}

	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (imageID + 1) << ": Hough space";
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	//scale
	cv::Mat temp8Bit;
	cv::normalize(flatHoughSpace, temp8Bit, 0, 255, cv::NORM_MINMAX);
	temp8Bit.convertTo(flatHoughSpace, CV_8U);
	//convert
	cv::imshow(windowName.str().c_str(), flatHoughSpace) ;

	//threshold hough space and display circles
	for (int i = 0; i < HOUGHY; ++i)
	{
		for (int j = 0; j < HOUGHX; ++j)
		{
			for (int r = 0; r < RMAX-RMIN; ++r)
			{
				if ( houghSpace[i][j][r] > HOUGHTHRESHOLD )
				{					
					circles.push_back( cv::Vec3d( i, j, r+RMIN ) ) ;
				}
			}
		}
	}

    //scaling
    double minx = *min_element( xes.begin(), xes.end() );
    double maxx = *max_element( xes.begin(), xes.end() );
    double miny = *min_element(yes.begin(), yes.end() );
    double maxy = *max_element(yes.begin(), yes.end() );

    std::vector<cv::Vec2d> centres ;
    bool putcircle = true ;

	while (!circles.empty())
	{
		cv::Vec3d tmp = circles.back() ;

		// fist we define the properties that the circle will have.
	    cv::Scalar redColour(255, 0, 0);
	    int radius = tmp[2];

	    // providing a negative number will create a filled circle
	    int thickness = 1;

	    // 8 connected line
	    // ( also there is a 4 connected line and CVAA which is an anti aliased line )
	    int linetype = 8; 

	    //recover old vslues
	    tmp[0] = (tmp[0] * (rowMax - rowMin) / (HOUGHY-1) ) -RMAX ; // (img.rows-1) / (HOUGHY-1) ;
	    tmp[1] = (tmp[1] * (colMax - colMin) / (HOUGHX-1) ) -RMAX ; // (img.cols-1) / (HOUGHX-1) ;

	    //rescale
	    tmp[0] = round ( ((img.rows) *(tmp[0] - miny))/(maxy-miny) );
	    tmp[1] = round ( ((img.cols) *(tmp[1] - minx))/(maxx-minx) );


	    //if similar circle was put dont do it again
	    for (std::vector<Vec2d>::iterator it = centres.begin() ; it != centres.end(); ++it)
	    {
	    	cv::Vec2d tmpc = *it ;
	    	if ( abs(tmpc[0] - tmp[0]) < INTERVAL && abs(tmpc[1] - tmp[1]) < INTERVAL )
	    	{
	    		putcircle = false ;
	    		break;
	    	 }
	    }

	    if ( putcircle )
	    {
	    	//std::cout << tmp[1] << " " << tmp[0] << std::endl;
	    	centres.push_back(cv::Vec2d(tmp[0], tmp[1])) ;
			// here is where we define the center of the circle
			cv::Point center( tmp[1], tmp[0] );
			cv::circle ( img , center , radius , redColour , thickness , linetype );
	    }
	    putcircle = true;

	    

		circles.pop_back() ;
	}
	windowName.str("");
	windowName.clear();
	windowName << "Coins " << (imageID + 1) << ": Detected Circles";
	cv::namedWindow(windowName.str().c_str(), CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName.str().c_str(), img) ;

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
		hough(i, gradTr, angle, coins[i]) ;
	}

	cv::waitKey();

	return 0;
}

