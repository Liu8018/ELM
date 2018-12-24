#include "elm_model.h"

ELM_Model::ELM_Model()
{
    m_I = -1;
    m_H = -1;
    m_O = -1;
    m_Q = -1;
    
    m_defaultActivationMethod = "sigmoid";
    m_randomState = -1;
}

void ELM_Model::inputData_2d(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool>> &labels, 
                             const int resizeWidth, const int resizeHeight, const int channels)
{
    m_channels = channels;
    m_width = resizeWidth;
    m_height = resizeHeight;
    
    //确定输入数据规模
    m_Q = mats.size();
    //确定输入层节点数
    m_I = m_width * m_height * m_channels;
    //确定输出层节点数
    m_O = labels[0].size();
    
    //转化label为target
    label2target(labels,m_Target);
    
    m_inputLayerData.create(cv::Size(m_I,m_Q),CV_32F);
    for(int i=0;i<mats.size();i++)
    {
        cv::Mat img;
        cv::resize(mats[i],img,cv::Size(m_width,m_height));
        
        cv::Mat lineROI = m_inputLayerData(cv::Range(i,i+1),cv::Range(0,m_inputLayerData.cols));
        mat2line(img,lineROI, m_channels);
    }
    normalize_img(m_inputLayerData);
    
//std::cout<<"m_Target:\n"<<m_Target<<std::endl;
//std::cout<<"m_inputLayerData:\n"<<m_inputLayerData<<std::endl;

}

void ELM_Model::inputData_2d_test(const std::vector<cv::Mat> &mats, const std::vector<std::vector<bool> > &labels)
{
    m_Q_test = mats.size();
    
    label2target(labels,m_Target_test);
    
    m_inputLayerData_test.create(cv::Size(m_I,m_Q_test),CV_32F);
    for(int i=0;i<mats.size();i++)
    {
        cv::Mat img;
        cv::resize(mats[i],img,cv::Size(m_width,m_height));
        
        cv::Mat lineROI = m_inputLayerData_test(cv::Range(i,i+1),cv::Range(0,m_inputLayerData_test.cols));
        mat2line(img,lineROI, m_channels);
    }
    normalize_img(m_inputLayerData_test);
}

void ELM_Model::setHiddenNodes(const int hiddenNodes)
{
    m_H = hiddenNodes;
}

void ELM_Model::setActivation(const std::string method)
{
    m_activationMethod = method;
}

void ELM_Model::setRandomState(int randomState)
{
    m_randomState = randomState;
}

void ELM_Model::fit()
{
    //检查隐藏层节点数是否被设置
    if(m_H == -1)
        m_H = m_Q/2;
    
    std::cout<<"Q:"<<m_Q<<std::endl;
    std::cout<<"I:"<<m_I<<std::endl;
    std::cout<<"H:"<<m_H<<std::endl;
    std::cout<<"O:"<<m_O<<std::endl;
    
    m_W_IH.create(cv::Size(m_H,m_I),CV_32F);
    m_W_HO.create(cv::Size(m_O,m_H),CV_32F);
    m_B_H.create(cv::Size(m_H,1),CV_32F);
    
    //第一步，随机产生IH权重和H偏置
    cv::RNG rng;
    if(m_randomState != -1)
        rng.state = m_randomState;
    else
        rng.state = (unsigned)time(NULL);
    for(int i=0;i<m_W_IH.rows;i++)
        for(int j=0;j<m_W_IH.cols;j++)
            m_W_IH.at<float>(i,j) = rng.uniform(-1.0,1.0);
    
    for(int j=0;j<m_B_H.cols;j++)
        m_B_H.at<float>(0,j) = rng.uniform(-1.0,1.0);
    
    //第二步，计算H输出
        //输入乘权重
    m_H_output = m_inputLayerData * m_W_IH;
        //加上偏置
    addBias(m_H_output,m_B_H);
        //激活
    if(m_activationMethod.empty())
        m_activationMethod = m_defaultActivationMethod;
    activate(m_H_output,m_activationMethod);
    
    //第三步，解出HO权重
    m_W_HO = m_H_output.inv(1) * m_Target;
    
/*std::cout<<"m_W_IH:\n"<<m_W_IH<<std::endl;
std::cout<<"m_B_H:\n"<<m_B_H<<std::endl;
std::cout<<"m_H_output:\n"<<m_H_output<<std::endl;
std::cout<<"m_W_HO:\n"<<m_W_HO<<std::endl;
std::cout<<"test:\n"<<m_H_output * m_W_HO<<"\n"<<m_Target<<std::endl;
*/
    //计算在训练数据上的准确率
    cv::Mat realOutput = m_H_output * m_W_HO;
    float finalScore = calcScore(realOutput,m_Target);
    std::cout<<"Score on training data:"<<finalScore<<std::endl;
    
    //计算在测试数据上的准确率
    if(!m_inputLayerData_test.empty())
    {
        cv::Mat m1 = m_inputLayerData_test * m_W_IH;
        addBias(m1,m_B_H);
        activate(m1,m_activationMethod);
        cv::Mat out = m1 * m_W_HO;
        float finalScore_test = calcScore(out,m_Target_test);
        
        std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
    }

}

void ELM_Model::addBias(cv::Mat &mat, const cv::Mat &bias)
{
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) += bias.at<float>(0,j);
}

void ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    //转化为一维数据
    cv::Mat inputLine(cv::Size(m_width*m_channels*m_height,1),CV_32F);
    cv::Mat tmpImg;
    cv::resize(mat,tmpImg,cv::Size(m_width,m_height));
    mat2line(tmpImg,inputLine,m_channels);
    normalize_img(inputLine);
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H,m_activationMethod);
    
    //计算输出
    cv::Mat output = H * m_W_HO;
    
    int id = getMaxId(output);
    label = m_label_string[id];
}

void ELM_Model::query(const cv::Mat &mat, cv::Mat &output)
{
    //转化为一维数据
    cv::Mat inputLine(cv::Size(m_width*m_channels*m_height,1),CV_32F);
    cv::Mat tmpImg;
    cv::resize(mat,tmpImg,cv::Size(m_width,m_height));
    mat2line(tmpImg,inputLine,m_channels);
    normalize_img(inputLine);
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H,m_activationMethod);
    
    //计算输出
    output = H * m_W_HO;
}

void ELM_Model::save(std::string path)
{
    cv::FileStorage fswrite(path,cv::FileStorage::WRITE);
    
    fswrite<<"channels"<<m_channels;
    fswrite<<"width"<<m_width;
    fswrite<<"height"<<m_height;
    fswrite<<"W_IH"<<m_W_IH;
    fswrite<<"W_HO"<<m_W_HO;
    fswrite<<"B_H"<<m_B_H;
    fswrite<<"activationMethod"<<m_activationMethod;
    fswrite<<"label_string"<<m_label_string;
    
    fswrite.release();
}

void ELM_Model::load(std::string path)
{
    cv::FileStorage fsread(path,cv::FileStorage::READ);
    
    fsread["channels"]>>m_channels;
    fsread["width"]>>m_width;
    fsread["height"]>>m_height;
    fsread["W_IH"]>>m_W_IH;
    fsread["W_HO"]>>m_W_HO;
    fsread["B_H"]>>m_B_H;
    fsread["activationMethod"]>>m_activationMethod;
    fsread["label_string"]>>m_label_string;
    
    fsread.release();
}

void ELM_Model::loadStandardDataset(const std::string datasetPath, const float trainSampleRatio,
                                    const int resizeWidth, const int resizeHeight, 
                                    const int channels, bool validate, bool shuffle)
{
    m_channels = channels;
    
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    
    inputImgsFrom(datasetPath,m_label_string,trainImgs,testImgs,trainLabelBins,testLabelBins,trainSampleRatio,channels,validate,shuffle);

    inputData_2d(trainImgs,trainLabelBins,resizeWidth,resizeHeight,channels);
    
    if(validate)
        inputData_2d_test(testImgs,testLabelBins);
}
