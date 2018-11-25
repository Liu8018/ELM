#include "elm_model.h"
#include <dirent.h>
#include <opencv2/highgui.hpp>

void traverseFile(const std::string directory, std::vector<std::string> &files)
{
    files.clear();
    
    const char * char_dir = directory.data();
    
    DIR* dir = opendir(char_dir);//打开指定目录
    dirent* p = NULL;//定义遍历指针
    while((p = readdir(dir)) != NULL)//开始逐个遍历
    {
        //linux平台下目录中有"."和".."隐藏文件，需要过滤掉
        if(p->d_name[0] != '.')//d_name是一个char数组，存放当前遍历到的文件名
        {
            std::string name = directory + std::string(p->d_name);
            files.push_back(name);
        }
    }
    closedir(dir);//关闭指定目录
}

int main()
{
    //读入数据集
    std::vector<std::string> files;
    traverseFile("/home/liu/下载/cnn-keras/dataset/bulbasaur/",files);
    
    std::vector<cv::Mat> imgs;
    std::vector<std::vector<bool>> labels;
    
    for(int i=0;i<files.size();i++)
    {
        cv::Mat src = cv::imread(files[i],0);
        imgs.push_back(src);
        std::vector<bool> label(2,0);
        label[0] = 1;
        labels.push_back(label);
    }
    
    traverseFile("/home/liu/下载/cnn-keras/dataset/charmander/",files);
    
    for(int i=0;i<files.size();i++)
    {
        cv::Mat src = cv::imread(files[i],0);
        imgs.push_back(src);
        std::vector<bool> label(2,0);
        label[1] = 1;
        labels.push_back(label);
    }
    
    ELM_Model model;
    model.inputData_2d(imgs,labels,50,50);
    model.fit();
    
    cv::Mat imgTest = cv::imread("/home/liu/下载/cnn-keras/examples/charmander_hidden.png",0);
    std::vector<bool> labelTest;
    model.query(imgTest,labelTest);
    
    /*std::vector<cv::Mat> mats;
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
        std::cout<<labelTest[i]<<std::endl;*/
    
    return 0;
}
