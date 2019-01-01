#include "delm_model.h"

DELM_Model::DELM_Model(int n_hiddenLayer, std::vector<int> hiddenNodes)
{
    m_n_hiddenLayer = n_hiddenLayer;
    m_hiddenNodes.assign(hiddenNodes.begin(),hiddenNodes.end());
}

void DELM_Model::loadStandardDataset(const std::string path, const float trainSampleRatio, 
                                           const int resizeWidth, const int resizeHeight, const int channels, bool shuffle)
{
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = channels;
    
    inputImgsFrom(path,m_label_string,m_trainImgs,
                  m_testImgs,m_trainLabelBins,m_testLabelBins,
                  trainSampleRatio,m_channels,shuffle);
}

void DELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle)
{
    loadMnistData_csv(path,trainSampleRatio,
                      m_trainImgs,m_testImgs,m_trainLabelBins,m_testLabelBins,shuffle);
    
    m_width = 28;
    m_height = 28;
    m_channels = 1;
}

void DELM_Model::fit()
{
    int I = m_width*m_height*m_channels;
    int Q = m_trainImgs.size();
    
    cv::Mat inputLayerData(cv::Size(I,Q),CV_32F);
    mats2lines(m_trainImgs,inputLayerData,m_channels);
    normalize_img(inputLayerData);
    cv::Mat Target;
    label2target(m_trainLabelBins,Target);
    
    cv::Mat inputLayerData_test;
    mats2lines(m_testImgs,inputLayerData_test,m_channels);
    normalize_img(inputLayerData_test);
    cv::Mat target_test;
    label2target(m_testLabelBins,target_test);
    
    int C = Target.cols;
    
    std::vector<cv::Mat> W(m_n_hiddenLayer);
    std::vector<cv::Mat> B(m_n_hiddenLayer);
    std::vector<cv::Mat> F(m_n_hiddenLayer);
    
    cv::Mat H = inputLayerData;
    
    //训练
    for(int i=0;i<m_n_hiddenLayer;i++)
    {
        if(i==0)
            randomGenerate(W[i],cv::Size(m_hiddenNodes[i],I));
        else
            randomGenerate(W[i],cv::Size(m_hiddenNodes[i],C));
        randomGenerate(B[i],cv::Size(m_hiddenNodes[i],1));
        
        H *= W[i];
        addBias(H,B[i]);
        activate(H,"sigmoid");
        
        F[i] = (H.t()*H).inv(1)*H.t()*Target;
        
        H *= F[i];
        
        float score = calcScore(H,Target);
        std::cout<<"training: score on "<<i<<" layer:"<<score<<std::endl;
    }
    
    //测试
    cv::Mat Ht = inputLayerData_test;
    
    for(int i=0;i<m_n_hiddenLayer;i++)
    {
        Ht *= W[i];
        addBias(Ht,B[i]);
        activate(Ht,"sigmoid");
        Ht *= F[i];
        
        float score = calcScore(Ht,target_test);
        std::cout<<"validating: score on "<<i<<" layer:"<<score<<std::endl;
    }
    
    //
    /*cv::Mat W1,W2,W3;
    cv::Mat B1,B2,B3;
    
    int h1=100,
        h2=100;
    
    for(int t=0;t<10;t++)
    {
        randomGenerate(W1,cv::Size(h1,I));
        randomGenerate(B1,cv::Size(h1,1));
        randomGenerate(W2,cv::Size(h2,C));
        randomGenerate(B2,cv::Size(h2,1));
        
        //训练第一层
        cv::Mat H = inputLayerData * W1;
        addBias(H,B1);
        activate(H,"sigmoid");
        
        cv::Mat F1 = (H.t()*H).inv(1)*H.t()*Target;
        
        //第一层validate
        H *= F1;
        
        cv::Mat Ht = inputLayerData_test * W1;
        addBias(Ht,B1);
        activate(Ht,"sigmoid");
        Ht *= F1;
        float score1 = calcScore(Ht,target_test);
        std::cout<<"layer 1 score on validation data:"<<score1<<std::endl;
        
        //训练第二层
        H *= W2;
        addBias(H,B2);
        activate(H,"sigmoid");
        
        cv::Mat F2 = (H.t()*H).inv(1)*H.t()*Target;
        
        //第二层validate
        H *= F2;
        
        Ht *= W2;
        addBias(Ht,B2);
        activate(Ht,"sigmoid");
        
        Ht *= F2;
        float score2 = calcScore(Ht,target_test);
        std::cout<<"layer 2 score on validation data:"<<score2<<std::endl;
        
    }*/
}

float DELM_Model::validate()
{
    
}
