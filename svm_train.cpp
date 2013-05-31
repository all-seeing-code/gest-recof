#include "opencv2/opencv.hpp"
#include "cv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sstream>
#include <dirent.h>

#define PATH "./gestures/"
using namespace std;
using namespace cv;

string path =PATH,train="./traindata/";
int count_gestures()
{
  DIR *dpdf;
  struct dirent *epdf;
  dpdf = opendir("./gestures");
  int count=0;
  if (dpdf != NULL)
  {
    while (epdf = readdir(dpdf))
    {
	count++;
    }
  }
  return count-2;
}


int count_files()
{
  DIR *dpdf;
  struct dirent *epdf;
  dpdf = opendir("./gestures/gesture1");
  int count=0;
  if (dpdf != NULL)
  {
    while (epdf = readdir(dpdf))
    {
	count++;
    }
  }
  return count-1;
}

/* To train our SVM on a set of images, first we have to construct the training matrix for the SVM. This matrix is specified as follows: each row of the matrix corresponds to one image, and each element in that row corresponds to one feature of the class -- in this case, the color of the pixel at a certain point. Since our images are 2D, we will need to convert them to a 1D matrix. The length of each row will be the area of the images (note that the images must be the same size).

Let's say we wanted to train the SVM on 5 different images, and each image was 4x3 pixels. First we would have to initialize the training matrix. The number of rows in the matrix would be 5, and the number of columns would be the area of the image, 4*3 = 12.
*/

int main()
{
  int no_files=count_files()-2;
  int no_gest=count_gestures();
  int img_area=32*24;  /* We have cropped the image to make it small enough. */
int i;
for(int nog=1;nog<=no_gest;nog++)
{
	 Mat training_mat(no_gest*no_files,img_area,CV_32FC1); /* Matrix generation for each gesture. */
  	int no_row=0;
	for(i=0;i<no_files;i++)
	{
		stringstream ss,ss1;
		ss<< nog;
		ss1<< i+1;
		Mat img_mat_1=imread(path+"/gesture"+ss.str()+"/gesture"+ss1.str()+".png",0);
		int ii=0;
		for(int k=0;k<img_mat_1.rows;k++)
		{
			for (int j = 0; j < img_mat_1.cols; j++) 
				{
						training_mat.at<float>(i,ii++) = img_mat_1.at<uchar>(k,j);
				}
		}
		no_row++;
	}
	
	for(int rest=1;rest<=no_gest;rest++)
	{
		if(rest==nog)continue;
		for(i=0;i<no_files;i++)
		{
			stringstream ss,ss1;
			ss<< rest;
			ss1<< i+1;
			Mat img_mat_1=imread(path+"/gesture"+ss.str()+"/gesture"+ss1.str()+".png",0);
			int ii=0;
			for(int k=0;k<img_mat_1.rows;k++)
			{
				for (int j = 0; j < img_mat_1.cols; j++) 
					{
							training_mat.at<float>(no_row,ii++) = img_mat_1.at<uchar>(k,j);
					}
			}
			no_row++;
		}
	}
		
		
		Mat labels(no_gest*no_files,1,CV_32FC1);
	  for(int i=0;i<=labels.rows;++i)
	  {
	  	if(i<no_files)
	    	labels.at<float>(i)=1;
	    else
	    	labels.at<float>(i)=-1;
	  }
	  
	  // Parameters to be set for SVM.
	  CvSVMParams params;
	  params.svm_type    = CvSVM::C_SVC;
	  params.kernel_type = CvSVM::LINEAR;
	  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	  CvSVM svm;
	  svm.train(training_mat, labels, Mat(), Mat(), params);
	  stringstream ss;
	  ss<< nog;
	  string name=(train+ss.str());
	  char * p;
	  p=&name[0];
	  svm.save(p);
	  //imwrite("./SVMGest"+ss.str()+".png",training_mat);
	}
	
   // saving
  /*svm.load("svm_filename"); // loading
  Mat img_mat_1 = imread("./lol.png",0);
  Mat img_mat_1d(1,32*24,CV_32FC1);
  int ii = 0; // Current column in training_mat
    for (int i = 0; i<img_mat_1.rows; i++) 
    {
      for (int j = 0; j < img_mat_1.cols; j++) 
	{
	    img_mat_1d.at<float>(0,ii++) = img_mat_1.at<uchar>(i,j);
	}
    }
  printf("%f\n",svm.predict(img_mat_1d));*/
  if(waitKey(0)>=0)
  return 0;
}
