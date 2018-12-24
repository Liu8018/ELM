#include "elm_cnn_model.h"

ELM_CNN_Model::ELM_CNN_Model()
{
    
}

void ELM_CNN_Model::test(const cv::Mat &inputData, cv::Mat &outputData)
{
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    kernel.at<float>(0,0) = 1.0;
    kernel.at<float>(2,0) = 2.0;
    kernel.at<float>(1,1) = 3.0;
    kernel.at<float>(0,2) = 4.0;
    kernel.at<float>(2,2) = 5.0;
    
    std::cout<<"kernel:\n"<<kernel<<std::endl;
    
    cv::filter2D(inputData,outputData,inputData.depth(),kernel,cv::Point(-1,-1),0,cv::BORDER_ISOLATED);
}
