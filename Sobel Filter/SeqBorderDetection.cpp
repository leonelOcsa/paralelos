#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdint.h>
#include <cstdlib>
#include <time.h> 
#include <stdio.h>
#include <chrono>

using namespace cv;
using namespace std;

int main()
{

	Mat kernel_1 = (Mat_<int>(3, 3) << -1, 0, 1,
									   -2, 0, 2,
									   -1, 0, 1);

	Mat kernel_2 = (Mat_<int>(3, 3) << -1, -2, -1,
									   0,  0,  0,
		                               1,  2,  1);

	Mat image, image2;
	image = imread("yoda.jpg", IMREAD_GRAYSCALE);   
	image2 = imread("yoda.jpg", CV_LOAD_IMAGE_COLOR);
	if (!image.data)                              
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	if (!image2.data)                              
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	namedWindow("Display window", WINDOW_AUTOSIZE);
	
	Mat grayimage; 

	cvtColor(image2, grayimage, CV_RGB2GRAY);

	image2 = Mat::zeros(image.size(), CV_8UC3);

	char* pixelPtr = (char*)image.data;
	char* pixelPtr2 = (char*)image2.data;
	int cn = image.channels();
	int cn2 = image2.channels();
	int R, G, B;


	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			cout << kernel_1.at<int>(i, j) << " ";
		}
		cout << endl;
	}

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			cout << kernel_2.at<int>(i, j) << " ";
		}
		cout << endl;
	}
	
	int count = 0;

	auto start_time = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < image.rows-3; i++){
		for (int j = 0; j < image.cols-3; j++){
			int sumB_1 = 0;
			int sumG_1 = 0;
			int sumR_1 = 0;
			int sumB_2 = 0;
			int sumG_2 = 0;
			int sumR_2 = 0;
			int sum_1 = 0;
			int sum_2 = 0;
			int media = 0;
			for (int k = 0; k < 3; k++){
				for (int l = 0; l < 3; l++){
					/*sumB_1 += kernel_1.at<int>(k,l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 0]);
					sumG_1 += kernel_1.at<int>(k, l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 1]);
					sumR_1 += kernel_1.at<int>(k, l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 2]);
					sumB_2 += kernel_2.at<int>(k, l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 0]);
					sumG_2 += kernel_2.at<int>(k, l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 1]);
					sumR_2 += kernel_2.at<int>(k, l)*((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 2]);
					*/
					//media = (int)(((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 0] + (int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 1] + (int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 2])/3);
					


					//sum_1 = kernel_1.at<int>(k, l)*media;
					//sum_2 = kernel_2.at<int>(k, l)*media;
					sum_1 += kernel_1.at<int>(k, l)*(int)image.at<uchar>(i + k, j + l);
					sum_2 += kernel_2.at<int>(k, l)*(int)image.at<uchar>(i + k, j + l);


					//pixelPtr2[(i)*image2.cols*cn2 + (j)*cn2 + 0] = 0;
					//pixelPtr2[(i)*image2.cols*cn2 + (j)*cn2 + 1] = 0;
					//pixelPtr2[(i)*image2.cols*cn2 + (j)*cn2 + 2] = 0;
					
					//if ( sqrt(sum_1*sum_1 + sum_2*sum_2) < 60){
						
					pixelPtr2[(i+1)*image2.cols*cn2 + (j+1)*cn2 + 0] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); //sumB_1 + sumB_2;
					pixelPtr2[(i+1)*image2.cols*cn2 + (j+1)*cn2 + 1] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); //sumG_1 + sumG_2;
					pixelPtr2[(i+1)*image2.cols*cn2 + (j+1)*cn2 + 2] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); //sumR_1 + sumR_2;
					
					//image.at<uchar>(Point(i + 1, j + 1)) = sqrt(sum_1*sum_1 + sum_2*sum_2);

					//image2.at<uchar>(i + 1, j + 1) = sqrt(sum_1*sum_1 + sum_2*sum_2);

						//}
					
						//if (count < 0){
					    //cout << (int)image.at<uchar>(i + k, j + l) << endl;

						//cout << ((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 0]) << " " << ((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 1]) << " " << ((int)pixelPtr[(i + k)*image.cols*cn + (j + l)*cn + 2]) << " " << (int)image.at<uchar>(i + k, j + l) << endl;


							//count++;
						//}

					

				}
			}
			//B = pixelPtr[i*image.cols*cn + j*cn + 0]; // B
			//G = pixelPtr[i*image.cols*cn + j*cn + 1]; // G
			//R = pixelPtr[i*image.cols*cn + j*cn + 2]; // R
		}
	}
	
	auto end_time = chrono::high_resolution_clock::now();
	auto time = end_time - start_time;

	cout << "El tiempo total fue de " <<chrono::duration_cast<chrono::milliseconds>(time).count() << ".\n";
	
	Canny(grayimage, grayimage, 100, 100 * 1.5, 3);

	imshow("Display window 1", image);               

	imshow("Canny", grayimage);

	imshow("Display window 2", image2);

	waitKey(0);                                         
	return 0;
}