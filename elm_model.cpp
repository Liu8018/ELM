#include "elm_model.h"

ELM_Model::ELM_Model()
{
    m_I = -1;
    m_H = -1;
    m_O = -1;
    m_Q = -1;
    
    m_defaultActivationMethod = "sigmoid";
}

void ELM_Model::inputData_2d(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, const int resizeWidth, const int resizeHeight)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    
    //确定输入数据规模
    m_Q = mats.size();
    //确定输入层节点数
    m_I = m_width * m_height;
    //确定输出层节点数
    m_O = labels[0].size();
    
    //检查隐藏层节点数是否被设置,若未设置则默认为与输入数据规模相等
    if(m_H == -1)
        m_H = m_Q;
    
    //转化label为target
    label2target(labels);
    
std::cout<<"m_Target:\n"<<m_Target<<std::endl;
    
    m_inputLayerData.create(cv::Size(m_I,m_Q),CV_32F);
    for(int i=0;i<mats.size();i++)
    {
        float * lineDataPtr = m_inputLayerData.ptr<float>(i);
        mat2line(mats[i],lineDataPtr);
    }
    
std::cout<<"m_inputLayerData:\n"<<m_inputLayerData<<std::endl;
}

void ELM_Model::mat2line(const cv::Mat &mat, float *lineDataPtr)
{
    cv::Mat img;
    cv::resize(mat,img,cv::Size(m_width,m_height));
    
    for(int r=0;r<img.rows;r++)
    {
        uchar * rowData = img.ptr<uchar>(r);
        for(int c=0;c<img.cols;c++)
        {
            lineDataPtr[r*img.cols+c] = float(rowData[c]);
        }
    }
}

void ELM_Model::setHiddenNodes(const int hiddenNodes)
{
    m_H = hiddenNodes;
}

void ELM_Model::setActivation(const std::string method)
{
    m_activationMethod = method;
}

void ELM_Model::label2target(const std::vector<std::vector<bool> > &labels)
{
    int labelLength = labels[0].size();
    m_Target.create(cv::Size(m_O,m_Q),CV_32F);
    for(int i=0;i<labels.size();i++)
    {
        for(int j=0;j<labelLength;j++)
            m_Target.at<float>(i,j) = float(labels[i][j]);
    }
}

void ELM_Model::fit()
{
    m_W_IH.create(cv::Size(m_H,m_I),CV_32F);
    m_W_HO.create(cv::Size(m_O,m_H),CV_32F);
    m_B_H.create(cv::Size(m_H,1),CV_32F);
    
    //第一步，随机产生IH权重和H偏置
    cv::RNG rng(8018);
    for(int i=0;i<m_W_IH.rows;i++)
    {
        float * W_rowData = m_W_IH.ptr<float>(i);
        for(int j=0;j<m_W_IH.cols;j++)
        {
            W_rowData[j] = rng.uniform(-1.0,1.0);
        }
    }
    
    {
        float * B_rowData = m_B_H.ptr<float>(0);
        for(int j=0;j<m_B_H.cols;j++)
        {
            B_rowData[j] = rng.uniform(-1.0,1.0);
        }
    }
    
    //第二步，计算H输出
        //输入乘权重
    m_H_output = m_inputLayerData * m_W_IH;
        //加上偏置
    addBias(m_H_output,m_B_H);
        //激活
    activate(m_H_output);
    
    //第三步，解出HO权重
    m_W_HO = m_H_output.inv(1) * m_Target;
    
std::cout<<"m_W_IH:\n"<<m_W_IH<<std::endl;
std::cout<<"m_B_H:\n"<<m_B_H<<std::endl;
std::cout<<"m_H_output:\n"<<m_H_output<<std::endl;
std::cout<<"m_W_HO:\n"<<m_W_HO<<std::endl;
std::cout<<"test:\n"<<m_H_output * m_W_HO<<std::endl;

}

void ELM_Model::addBias(cv::Mat &mat, const cv::Mat &bias)
{
    for(int i=0;i<mat.rows;i++)
    {
        float * H_O_rowData = mat.ptr<float>(i);
        const float * B_rowData = bias.ptr<const float>(0);
        
        for(int j=0;j<mat.cols;j++)
        {
            H_O_rowData[j] += B_rowData[j];
        }
    }
}

void ELM_Model::activate(cv::Mat &H)
{
    if(m_activationMethod == "sigmoid")
        sigmoid(H);
    else
    {
        m_activationMethod = m_defaultActivationMethod;
        activate(H);
    }
}

void ELM_Model::sigmoid(cv::Mat &H)
{
    for(int i=0;i<H.rows;i++)
    {
        float * H_rowData = H.ptr<float>(i);
        for(int j=0;j<H.cols;j++)
        {
            H_rowData[j] = 1 / ( 1 + std::exp(-H_rowData[j]) );
        }
    }
}

void ELM_Model::query(const cv::Mat &mat, std::vector<bool> &label)
{
    //转化为一维数据
    cv::Mat inputImg;
    cv::resize(mat,inputImg,cv::Size(m_width,m_height));
    cv::Mat inputLine(cv::Size(m_width*m_height,1),CV_32F);
    float * lineDataPtr = inputLine.ptr<float>(0);
    mat2line(mat,lineDataPtr);
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H);
    
    //计算输出
    cv::Mat output = H * m_W_HO;
    
    //转化为二进制label
    label.clear();
    for(int j=0;j<output.cols;j++)
    {
        if(output.at<float>(0,j) >= 0.5)
            label.push_back(1);
        else
            label.push_back(0);
    }
}
