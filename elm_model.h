#ifndef ELM_MODEL_H
#define ELM_MODEL_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>

class ELM_Model
{
public:
    ELM_Model();
    
    //设置隐藏层节点数
    void setHiddenNodes(const int hiddenNodes);
    //输入二维数据
    void inputData_2d(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, const int resizeWidth, const int resizeHeight);
    //设置激活函数
    void setActivation(const std::string method);
    
    //
    void fit();
    
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
    
    //转化label为target
    void label2target(const std::vector<std::vector<bool>> &labels);
    
    //激活
    void activate(cv::Mat &H);
    std::string m_activationMethod;
    std::string m_defaultActivationMethod;
    void sigmoid(cv::Mat &H);
    
};

#endif // ELM_MODEL_H