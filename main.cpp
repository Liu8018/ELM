#include "elm_model.h"

int main()
{
    std::vector<cv::Mat> mats;
    cv::Mat a(cv::Size(3,2),CV_8U);
    for(int i=0;i<a.rows;i++)
        for(int j=0;j<a.cols;j++)
            a.at<uchar>(i,j) = i+j;
    cv::Mat b(cv::Size(3,2),CV_8U);
    for(int i=0;i<b.rows;i++)
        for(int j=0;j<b.cols;j++)
            b.at<uchar>(i,j) = 2*i+j;
    cv::Mat c(cv::Size(3,2),CV_8U);
    for(int i=0;i<c.rows;i++)
        for(int j=0;j<c.cols;j++)
            c.at<uchar>(i,j) = i+2*j;
    cv::Mat d(cv::Size(3,2),CV_8U);
    for(int i=0;i<d.rows;i++)
        for(int j=0;j<d.cols;j++)
            d.at<uchar>(i,j) = i*j;
    
    mats.push_back(a);
    mats.push_back(b);
    mats.push_back(c);
    mats.push_back(d);
    
    std::cout<<a<<std::endl;
    std::cout<<b<<std::endl;
    std::cout<<c<<std::endl;
    std::cout<<d<<std::endl;
    
    std::vector<std::vector<bool>> labels;
    std::vector<bool> label1(2,0);
    std::vector<bool> label2(2,1);
    std::vector<bool> label3(2);label3[0] = 0;label3[1] = 1;
    
    labels.push_back(label1);
    labels.push_back(label2);
    labels.push_back(label3);
    labels.push_back(label3);
    
    ELM_Model model;
    model.inputData_2d(mats,labels,3,2);
    model.fit();
    
    d.at<uchar>(1,2) = 3;
    std::vector<bool> labelTest;
    model.query(d,labelTest);
    for(int i=0;i<labelTest.size();i++)
        std::cout<<labelTest[i]<<std::endl;
    
    
    return 0;
}
