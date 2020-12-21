#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/core/utility.hpp>
using namespace cv;
using namespace std;

int main( int argc, char** argv ){

  if(argc<2){
    std::cout << "Indicar una imagen como parámetro";
    exit(1);
  }

  cv::Mat imagen = imread(argv[1], cv::IMREAD_COLOR);
  if(imagen.empty()){
    std::cout << "Indicar una imagen como parámetro";
    exit(1);
  }

  imshow("Imagen original",imagen);

  int h = imagen.rows;
  int w = imagen.cols;
  
  Mat filtro = Mat::zeros(h, w, CV_8UC3); 
  medianBlur(imagen,filtro,3);

  imshow("Filtro de la mediana",filtro);

  Mat gray; Mat dst; 
  Mat bordes = Mat::zeros(h, w, CV_8UC3);
  cvtColor( filtro, gray, COLOR_BGR2GRAY );

  Mat bordescanny = Mat::zeros(h, w, CV_8UC3);
  Canny(gray, bordescanny,60,80);
  convertScaleAbs( bordescanny, bordescanny );

  for(int i=0; i<h; i++){
    for(int j=0; j<w; j++){
      int intensidad = (int)bordescanny.at<uchar>(i,j);
      if(intensidad<45)
        bordescanny.at<uchar>(i,j) = 255;
      else
        bordescanny.at<uchar>(i,j) = 0;
    }
  }

  imwrite("canny.png", bordescanny);
  imshow( "canny", bordescanny );

  Mat labels, centers;
  int valores_centrales;
  
  Mat filtro2 = Mat::zeros(h, w, CV_8UC3); 

  filtro.convertTo(filtro2,CV_32F);
  filtro2 = filtro2.reshape(1,filtro2.total());
  kmeans(filtro2,8,labels,TermCriteria( 1000, 10, 1.0),10,KMEANS_PP_CENTERS,centers);
  
  centers = centers.reshape(3,centers.rows);
  filtro2 = filtro2.reshape(3,filtro2.rows);

  Vec3f *pixel = filtro2.ptr<Vec3f>();

  for (size_t i=0; i<filtro2.rows; i++) {
  valores_centrales = labels.at<int>(i);
  pixel[i] = centers.at<Vec3f>(valores_centrales);
  }

  filtro = filtro2.reshape(3, filtro.rows);
  filtro.convertTo(filtro, CV_8UC3);

  imshow("Reducción de cantidad de colores ",filtro);

  cv::cvtColor(bordescanny,bordescanny,COLOR_GRAY2BGR);

  for(int i=0; i<h; i++){
    for(int j=0; j<w*bordescanny.channels(); j++){
      int intensidad = (int)bordescanny.at<uchar>(i,j);
      if(intensidad==0)
        filtro.at<uchar>(i,j) = 0;
    }
  }
  imwrite("cannyfinal.png",filtro);
  imshow("Recombinación de bordes con reducción de bordes",filtro);

  waitKey();
  return 0;
}