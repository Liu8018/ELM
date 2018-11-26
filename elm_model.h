#ifndef ELM_MODEL_H
#define ELM_MODEL_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <dirent.h>
#include <algorithm>

class ELM_Model
{
public:
    ELM_Model();
    
    //设置隐藏层节点数
    void setHiddenNodes(const int hiddenNodes);
    //输入二维数据
    void inputData_2d(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, 
                      const int resizeWidth, const int resizeHeight, const int channels);
    void inputData_2d_test(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels);
    //设置激活函数
    void setActivation(const std::string method);
    
    //
    void fit();
    
    //查询
    void query(const cv::Mat &mat, std::vector<bool> &label);
    
    //保存和读取模型
    void save(std::string path);
    void load(std::string path);
    
    void loadStandardDataset(const std::string datasetPath, const float testSampleRatio,
                             const int resizeWidth, const int resizeHeight, const int channels);
    
private:
    int m_I;  //输入层节点数
    int m_H;  //隐藏层节点数
    int m_O;  //输出层节点数
    int m_Q;  //输入数据规模
    
    //三层
    cv::Mat m_inputLayerData;   //m_Q×m_I
    cv::Mat m_H_output;         //m_Q×m_H
    cv::Mat m_Target;           //m_Q×m_O
    
    //权重
    cv::Mat m_W_IH;     //m_I×m_H
    cv::Mat m_W_HO;     //m_H×m_O
    //偏置
    cv::Mat m_B_H;      //1×m_H
    
    //二维数据转换为一维
    void mat2line(const cv::Mat &mat, cv::Mat &line);
    
    //加偏置
    void addBias(cv::Mat &mat, const cv::Mat &bias);
    
    //转化label为target
    void label2target(const std::vector<std::vector<bool>> &labels, cv::Mat &target);
    
    //激活
    void activate(cv::Mat &H);
    std::string m_activationMethod;
    std::string m_defaultActivationMethod;
    void sigmoid(cv::Mat &H);
    
    //归一化
    void normalize(cv::Mat &mat);
    
    int m_channels;
    int m_width;
    int m_height;
    
    std::vector<std::string> m_label_string;
    
    void traverseFile(const std::string directory, std::vector<std::string> &files);
    
    int m_Q_test;
    cv::Mat m_Target_test;
    cv::Mat m_inputLayerData_test;
    
    float calcScore(const cv::Mat &outputData, const cv::Mat &target);
};

#endif // ELM_MODEL_H
