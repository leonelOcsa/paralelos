
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <stdint.h>
#include <cstdlib>
#include <time.h> 

using namespace cv;
using namespace std;

__global__ void apply_mask(uchar *image, uchar *imageX, int *mask_1, int *mask_2, int rows, int cols, int mask_size){
	//los offset conforman los saltos de iteración dado por la multiplicación del total de bloques de un lado por el numero de threads de un lado del bloque 
	int offset_x = gridDim.x*blockDim.x; 
	int offset_y = gridDim.y*blockDim.y;

	/*
	printf("X: %d - %d \n", idx, threadIdx.x);
	printf("Y: %d - %d \n", idy, threadIdx.y);
	*/

	for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < rows; i += offset_x){ //cada threadIdx contiene una id unica que va de 0 a n, segun la dimension de threads por bloque en una dirección
		for (int j = blockIdx.y*blockDim.y + threadIdx.y; j < cols; j += offset_y){
			int sum_1 = 0, sum_2 = 0; //almacenaran los valores de aplicar las mascaras 1 y 2 de sobel
			for (int k = 0; k < 3; k++){
				for (int l = 0; l < 3; l++){
					sum_1 += image[(i + k)*rows + (j + l)] * mask_1[k * 3 + l];
					sum_2 += image[(i + k)*rows + (j + l)] * mask_2[k * 3 + l];
				}
			}
			//Mediante la distancia euclidea procedemos a reemplazar el punto central en la imagen resultante
			imageX[(i + 1)*rows + (j + 1)] = sqrtf(sum_1*sum_1 + sum_2*sum_2); 
		}	
	}

	/*
	int begin_x = idx + threadIdx.x; //inicio de la thread en x
	//int end_x = begin_x + mask_size;
	int end_x = begin_x + blockDim.x;
	int begin_y = idy + threadIdx.y; //inicio de la thread en y
	//int end_y = begin_y + mask_size;
	int end_y = begin_y + blockDim.y;

	*/
}

int main()
{   
	Mat image, image2, image3;
	image = imread("yoda.jpg", IMREAD_GRAYSCALE);
	image2 = imread("yoda.jpg", CV_LOAD_IMAGE_COLOR);
	image3 = imread("yoda.jpg", CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	//imagen representado por un vector en Host
	uchar *img = image.data;
	int rows = image.rows;
	int cols = image.cols;
	int imageFullSize = rows*cols;

	float time;
	
	//imagen representado por un vector en Device
	size_t d_img_size = imageFullSize * sizeof(uchar);
	uchar * d_img, *d_img_x;
	cudaMalloc((void**)&d_img, d_img_size);
	cudaMalloc((void**)&d_img_x, d_img_size);
	cudaMemcpy(d_img, img, d_img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_x, img, d_img_size, cudaMemcpyHostToDevice);

	//tiempo en GPU
	cudaEvent_t start, stop;

	dim3 dimBlock(4, 4); //dimension de los bloques
	dim3 dimThread(6, 6); //threads por bloque

	//creamos los vector que almacenarán los valores de ambas máscaras en el host
	int maskSize = 9;
	size_t m_size = maskSize * sizeof(int);
	int *mask_1 = (int*)malloc(m_size);
	int *mask_2 = (int*)malloc(m_size);
	int *mask_prueba = (int*)malloc(m_size);

	//ahora inicializo ambas máscaras con sus valores respectivos
	mask_1[0] = -1; mask_1[1] = 0; mask_1[2] = 1;
	mask_1[3] = -2; mask_1[4] = 0; mask_1[5] = 2;
	mask_1[6] = -1; mask_1[7] = 0; mask_1[8] = 1;

	//ahora inicializo ambas máscaras con sus valores respectivos
	mask_2[0] = -1; mask_2[1] = -2; mask_2[2] = -1;
	mask_2[3] = 0; mask_2[4] = 0; mask_2[5] = 0;
	mask_2[6] = 1; mask_2[7] = 2; mask_2[8] = 1;

	//creamos los vectores que almacenarán los valores de ambas máscaras en el device
	int *dmask_1, *dmask_2;
	cudaMalloc((void**)&dmask_1, m_size);
	cudaMalloc((void**)&dmask_2, m_size);

	//pasamos las máscaras del host al device
	cudaMemcpy(dmask_1, mask_1, m_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dmask_2, mask_2, m_size, cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	apply_mask <<< dimBlock, dimThread >>> (d_img, d_img_x, dmask_1, dmask_2, rows, cols, maskSize);

	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\nTiempo de procesamiento: %f ms\n", time);

	cudaMemcpy(mask_prueba, dmask_1, m_size, cudaMemcpyDeviceToHost);
	
	uchar *h_img_x = (uchar *)malloc(d_img_size);
	//copiamos el array del device al host
	cudaMemcpy(h_img_x, d_img_x, d_img_size, cudaMemcpyDeviceToHost);
	
	char* pixelPtr3 = (char*)image3.data;
	int cn3 = image3.channels();
	//aca procedo a convertir mi array a un Mat
	cv::Mat TempMat = cv::Mat(image3.rows, image3.cols, CV_8UC1, h_img_x);
	
	// Liberamos la memoria del Host
	free(mask_1);
	free(mask_2);
	free(mask_prueba);

	// Liberamos la memoria del Device
	cudaFree(dmask_1);
	cudaFree(dmask_2);

	//Version secuencial en CPU

	Mat kernel_1 = (Mat_<int>(3, 3) << -1, 0, 1,
	-2, 0, 2,
	-1, 0, 1);

	Mat kernel_2 = (Mat_<int>(3, 3) << -1, -2, -1,
		0, 0, 0,
		1, 2, 1);

	

	namedWindow("Display window", WINDOW_AUTOSIZE);

	image2 = Mat::zeros(image.size(), CV_8UC3);
	
	char* pixelPtr = (char*)image.data;
	char* pixelPtr2 = (char*)image2.data;
	int cn = image.channels();
	int cn2 = image2.channels();
	int R, G, B;


	for (int i = 0; i < image.rows - 3; i++){
		for (int j = 0; j < image.cols - 3; j++){
			int sum_1 = 0;
			int sum_2 = 0;
			for (int k = 0; k < 3; k++){
				for (int l = 0; l < 3; l++){
					sum_1 += kernel_1.at<int>(k, l)*image.at<uchar>(i + k, j + l);
					sum_2 += kernel_2.at<int>(k, l)*image.at<uchar>(i + k, j + l);

					pixelPtr2[(i + 1)*image2.cols*cn2 + (j + 1)*cn2 + 0] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); 
					pixelPtr2[(i + 1)*image2.cols*cn2 + (j + 1)*cn2 + 1] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); 
					pixelPtr2[(i + 1)*image2.cols*cn2 + (j + 1)*cn2 + 2] = (int)sqrt(sum_1*sum_1 + sum_2*sum_2); 
				}
			}
		}
	}

	imshow("Imagen de entrada - Escala de Grises", image);
	imshow("Versión Secuencial CPU", image2);
	imshow("Versión Paralela GPU", TempMat);
	
	waitKey(0);
	
	return 0;
}
