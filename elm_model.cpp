#include "elm_model.h"

ELM_Model::ELM_Model()
{
    m_I = -1;
    m_H = -1;
    m_O = -1;
    m_Q = -1;
    
    m_defaultActivationMethod = "sigmoid";
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
        cv::Mat lineROI = m_inputLayerData(cv::Range(i,i+1),cv::Range(0,m_inputLayerData.cols));
        mat2line(mats[i],lineROI);
    }
    
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
        cv::Mat lineROI = m_inputLayerData_test(cv::Range(i,i+1),cv::Range(0,m_inputLayerData_test.cols));
        mat2line(mats[i],lineROI);
    }
}

void ELM_Model::mat2line(const cv::Mat &mat, cv::Mat &line)
{
    cv::Mat img;
    cv::resize(mat,img,cv::Size(m_width,m_height));
    
    //cv::imshow("resizedImg",img);
    //cv::waitKey();
    
    if(m_channels==1)
    {
        for(int r=0;r<img.rows;r++)
            for(int c=0;c<img.cols;c++)
                line.at<float>(0,r*img.cols+c) = float(img.at<uchar>(r,c));
    }
    if(m_channels==3)
    {
        std::vector<cv::Mat> channels;
        cv::split(img,channels);
        int j=0;
        for(int i=0;i<3;i++)
            for(int r=0;r<channels[i].rows;r++)
                for(int c=0;c<channels[i].cols;c++)
                {
                    line.at<float>(0,j) = float(channels[i].at<uchar>(r,c));
                    j++;
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

void ELM_Model::label2target(const std::vector<std::vector<bool> > &labels, cv::Mat &target)
{
    int labelLength = labels[0].size();
    target.create(cv::Size(m_O,labels.size()),CV_32F);
    for(int i=0;i<labels.size();i++)
    {
        for(int j=0;j<labelLength;j++)
            target.at<float>(i,j) = float(labels[i][j]);
    }
}

void ELM_Model::fit()
{
    //检查隐藏层节点数是否被设置
    if(m_H == -1)
        m_H = m_Q/2;
    
    m_W_IH.create(cv::Size(m_H,m_I),CV_32F);
    m_W_HO.create(cv::Size(m_O,m_H),CV_32F);
    m_B_H.create(cv::Size(m_H,1),CV_32F);
    
    //第一步，随机产生IH权重和H偏置
    cv::RNG rng((unsigned)time(NULL));
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
    activate(m_H_output);
    
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
        activate(m1);
        cv::Mat out = m1 * m_W_HO;
        float finalScore_test = calcScore(out,m_Target_test);
        
        std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
    }

}

float ELM_Model::calcScore(const cv::Mat &outputData, const cv::Mat &target)
{
    int score = 0;
    for(int i=0;i<outputData.rows;i++)
    {
        cv::Mat ROI_o = outputData(cv::Range(i,i+1),cv::Range(0,outputData.cols));
        int maxId_O = getMaxId(ROI_o);
        
        cv::Mat ROI_t = target(cv::Range(i,i+1),cv::Range(0,target.cols));
        int maxId_T = getMaxId(ROI_t);
        
        if(maxId_O == maxId_T)
            score++;
    }
std::cout<<"score:"<<score<<std::endl;
std::cout<<"outputData.rows:"<<outputData.rows<<std::endl;
    float finalScore = score/(float)outputData.rows;
    
    return finalScore;
}

int ELM_Model::getMaxId(const cv::Mat &line)
{
    double minVal,maxVal;
    int minIdx[2],maxIdx[2];
    
    cv::minMaxIdx(line,&minVal,&maxVal,minIdx,maxIdx);
    
    return maxIdx[1];
}

void ELM_Model::addBias(cv::Mat &mat, const cv::Mat &bias)
{
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) += bias.at<float>(0,j);
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
        for(int j=0;j<H.cols;j++)
            H.at<float>(i,j) = 1 / ( 1 + std::exp(-H.at<float>(i,j)) );
}

void ELM_Model::normalize(cv::Mat &mat)
{
    double minVal,maxVal;
    cv::minMaxIdx(mat,&minVal,&maxVal);
    
    for(int i=0;i<mat.rows;i++)
        for(int j=0;j<mat.cols;j++)
            mat.at<float>(i,j) = (mat.at<float>(i,j)-minVal) / (maxVal-minVal);
}

void ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    //转化为一维数据
    cv::Mat inputLine(cv::Size(m_width*m_channels*m_height,1),CV_32F);
    mat2line(mat,inputLine);
    
    //乘权重，加偏置，激活
    cv::Mat H = inputLine * m_W_IH;
    addBias(H,m_B_H);
    activate(H);
    
    //计算输出
    cv::Mat output = H * m_W_HO;
std::cout<<"output:\n"<<output<<std::endl;
    normalize(output);
std::cout<<"normalized output:\n"<<output<<std::endl;
    
    int id = getMaxId(output);
    label = m_label_string[id];
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

void ELM_Model::traverseFile(const std::string directory, std::vector<std::string> &files)
{
    std::string prefix = directory;
    if(directory[directory.length()-1] != '/')
        prefix += '/';
    
    files.clear();
    
    const char * char_dir = directory.data();
    
    DIR* dir = opendir(char_dir);//打开指定目录
    dirent* p = NULL;//定义遍历指针
    while((p = readdir(dir)) != NULL)//开始逐个遍历
    {
        //linux平台下目录中有"."和".."隐藏文件，需要过滤掉
        if(p->d_name[0] != '.')//d_name是一个char数组，存放当前遍历到的文件名
        {
            std::string name = prefix + std::string(p->d_name);
            files.push_back(name);
        }
    }
    closedir(dir);//关闭指定目录
}

void ELM_Model::loadStandardDataset(const std::string datasetPath, const float testSampleRatio,
                                    const int resizeWidth, const int resizeHeight, const int channels)
{
    m_channels = channels;
    
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    
    std::vector<std::string> files;
    traverseFile(datasetPath,files);
    
    int classes = files.size();
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    
    for(int i=0;i<files.size();i++)
    {
        std::cout<<"[INFO] loading data from "<<files[i]<<std::endl;
        
        std::vector<std::string> subdir_files;
        traverseFile(files[i],subdir_files);
        std::random_shuffle(subdir_files.begin(),subdir_files.end());
        
        if(!subdir_files.empty())
        {
            std::string label = files[i].substr(files[i].find_last_of('/')+1,files[i].length()-1);
            m_label_string.push_back(label);
            
            int trainSamples = subdir_files.size()*(1.0-testSampleRatio);
            
            for(int j=0;j<trainSamples;j++)
            {
                //std::cout<<subdir_files[j]<<std::endl;
                
                cv::Mat src;
                if(channels == 3)
                    src = cv::imread(subdir_files[j]);
                if(channels == 1)
                    src = cv::imread(subdir_files[j],0);
                
                trainImgs.push_back(src);
                
                std::vector<bool> labelBin(classes,0);
                labelBin[i] = 1;
                trainLabelBins.push_back(labelBin);
            }
            
            for(int j=trainSamples;j<subdir_files.size();j++)
            {
                //std::cout<<subdir_files[j]<<std::endl;
                
                cv::Mat src;
                if(channels == 3)
                    src = cv::imread(subdir_files[j]);
                if(channels == 1)
                    src = cv::imread(subdir_files[j],0);
                
                testImgs.push_back(src);
                
                std::vector<bool> labelBin(classes,0);
                labelBin[i] = 1;
                testLabelBins.push_back(labelBin);
            }
        }
    }

    inputData_2d(trainImgs,trainLabelBins,resizeWidth,resizeHeight,channels);
    
    if(testSampleRatio > 0)
        inputData_2d_test(testImgs,testLabelBins);
}
