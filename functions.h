#ifndef FUNCS_H
#define FUNCS_H

#endif // FUNCS_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dirent.h>
#include <algorithm>

//加载图像
void inputImgsFrom(const std::string datasetPath, 
                   std::vector<std::string> &label_string,
                   std::vector<cv::Mat> &trainImgs, std::vector<cv::Mat> &testImgs, 
                   std::vector<std::vector<bool> > &trainLabelBins, 
                   std::vector<std::vector<bool> > &testLabelBins, 
                   const float trainSampleRatio, const int channels, 
                   bool validate=true, bool shuffle=true);

//二维数据转换为一维(从AxB到1xAB)
void mat2line(const cv::Mat &mat, cv::Mat &line, const int channels);

//转化label为target(从QxC到QxC)
void label2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target);

//激活
void activate(cv::Mat &H, const std::string method);

//激活函数sigmoid
void sigmoid(cv::Mat &H);

//归一化
void normalize(cv::Mat &mat);

//遍历一个目录
void traverseFile(const std::string directory, std::vector<std::string> &files);

int getMaxId(const cv::Mat &line);

float calcScore(const cv::Mat &outputData, const cv::Mat &target);
