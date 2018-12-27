#include "elm_in_elm_model.h"

ELM_IN_ELM_Model::ELM_IN_ELM_Model(const int n_models, const std::string modelDir)
{
    m_n_models = n_models;
    m_subModelHiddenNodes.resize(m_n_models);
    
    for(int i=0;i<m_n_models;i++)
        m_subModelHiddenNodes[i] = -1;
    
    m_modelPath = modelDir;
    if(m_modelPath[m_modelPath.length()-1] != '/')
        m_modelPath.append("/");
}

void ELM_IN_ELM_Model::setSubModelHiddenNodes(const int modelId, const int n_nodes)
{
    if(modelId == -1)
    {
        for(int i=0;i<m_n_models;i++)
            m_subModelHiddenNodes[i] = n_nodes;
    }
    else
        m_subModelHiddenNodes[modelId] = n_nodes;
}

void ELM_IN_ELM_Model::loadStandardDataset(const std::string path, const float trainSampleRatio, 
                                           const int resizeWidth, const int resizeHeight, const int channels, 
                                           bool validate, bool shuffle)
{
    m_datasetPath = path;
    m_trainSampleRatio = trainSampleRatio;
    m_width = resizeWidth;
    m_height = resizeHeight;
    m_channels = channels;
    m_validate = validate;
    m_shuffle = shuffle;
    
    m_subModelToTrain.loadStandardDataset(m_datasetPath,m_trainSampleRatio,m_width,m_height,m_channels,m_validate,m_shuffle);
}

void ELM_IN_ELM_Model::loadMnistData(const std::string path, const float trainSampleRatio, bool validate, bool shuffle)
{
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    loadMnistData_csv(path,trainSampleRatio,trainImgs,testImgs,trainLabelBins,testLabelBins,validate,shuffle);
    
    m_subModelToTrain.inputData_2d(trainImgs,trainLabelBins,28,28,1);
    if(validate)
        m_subModelToTrain.inputData_2d_test(testImgs,testLabelBins);
}

void ELM_IN_ELM_Model::fitSubModels(int batchSize)
{
    int randomState = (unsigned)time(NULL);
    
    //训练子模型
    for(int i=0;i<m_n_models;i++)
    {
        if(m_subModelHiddenNodes[i] != -1)
            m_subModelToTrain.setHiddenNodes(m_subModelHiddenNodes[i]);
        m_subModelToTrain.setRandomState(randomState++);
        m_subModelToTrain.fit(batchSize);
        m_subModelToTrain.save(m_modelPath+"subModel"+std::to_string(i)+".xml",
                               m_modelPath+"subK"+std::to_string(i)+".xml");
    }
}

void ELM_IN_ELM_Model::fitMainModel(int Q)
{
    //载入子模型
    std::vector<ELM_Model> subModels;
    subModels.resize(m_n_models);
    for(int i=0;i<m_n_models;i++)
        subModels[i].load(m_modelPath+"subModel"+std::to_string(i)+".xml",
                          m_modelPath+"subK"+std::to_string(i)+".xml");
    
    //载入训练数据
    std::vector<cv::Mat> trainImgs;
    std::vector<cv::Mat> testImgs;
    std::vector<std::vector<bool>> trainLabelBins;
    std::vector<std::vector<bool>> testLabelBins;
    inputImgsFrom(m_datasetPath,m_label_string,trainImgs,testImgs,trainLabelBins,testLabelBins,m_trainSampleRatio,m_channels,m_validate,m_shuffle);
    
    if(int(trainImgs.size()) < Q)
    {
        std::cout<<"Too small scale of training data!"<<std::endl;
        return;
    }
    
    //为H和T分配空间
    int M = m_n_models;
    if(Q==-1)
        m_Q = trainImgs.size();
    else
        m_Q = Q;
    m_C = trainLabelBins[0].size();
    cv::Mat H(cv::Size(M*m_C,m_Q),CV_32F);
    cv::Mat T(cv::Size(m_C,m_Q),CV_32F);
    
    std::cout<<"Q:"<<m_Q<<std::endl
             <<"M:"<<M<<std::endl
             <<"C:"<<m_C<<std::endl;
    
    //为H和T赋值
    for(int q=0;q<m_Q;q++)
    {
        for(int m=0;m<M;m++)
        {
            cv::Mat ROI = H(cv::Range(q,q+1),cv::Range(m*m_C,(m+1)*m_C));
            subModels[m].query(trainImgs[q],ROI);
            normalize(ROI);
        }
        
        for(int c=0;c<m_C;c++)
            T.at<float>(q,c) = float(trainLabelBins[q][c]);
    }
    
    //求解F
    m_F = H.inv(1) * T;
    
    //计算在训练数据上的准确率
    cv::Mat realOutput = H * m_F;
    float finalScore = calcScore(realOutput,T);
    std::cout<<"Score on training data:"<<finalScore<<std::endl;
    
/*std::cout<<"T:"<<T.size<<"\n"<<T<<std::endl;
std::cout<<"H:"<<H.size<<"\n"<<H<<std::endl;
std::cout<<"F:"<<m_F.size<<"\n"<<m_F<<std::endl;
std::cout<<"H*F:"<<realOutput.size<<"\n"<<H*m_F<<std::endl;
*/
    //计算在测试数据上的准确率
    if(m_validate)
    {
        cv::Mat H_test(cv::Size(M*m_C,testImgs.size()),CV_32F);
        cv::Mat T_test(cv::Size(m_C,testImgs.size()),CV_32F);
        
        //给H_test和T_test赋值
        for(int q=0;q<int(testImgs.size());q++)
        {
            for(int m=0;m<M;m++)
            {
                cv::Mat ROI = H_test(cv::Range(q,q+1),cv::Range(m*m_C,(m+1)*m_C));
                subModels[m].query(testImgs[q],ROI);
                normalize(ROI);
            }
            
            for(int c=0;c<m_C;c++)
                T_test.at<float>(q,c) = float(testLabelBins[q][c]);
        }
        
        //计算
        cv::Mat output = H_test * m_F;
        float finalScore_test = calcScore(output,T_test);
        
        std::cout<<"Score on validation data:"<<finalScore_test<<std::endl;
    }
}

void ELM_IN_ELM_Model::save()
{
    cv::FileStorage fswrite(m_modelPath+"mainModel.xml",cv::FileStorage::WRITE);
    
    fswrite<<"n_models"<<m_n_models;
    fswrite<<"subModelPath"<<m_modelPath;
    fswrite<<"channels"<<m_channels;
    fswrite<<"C"<<m_C;
    fswrite<<"F"<<m_F;
    fswrite<<"label_string"<<m_label_string;
    
    fswrite.release();
}

void ELM_IN_ELM_Model::load()
{
    cv::FileStorage fsread(m_modelPath+"mainModel.xml",cv::FileStorage::READ);

    fsread["n_models"]>>m_n_models;
    fsread["subModelPath"]>>m_modelPath;
    fsread["channels"]>>m_channels;
    fsread["C"]>>m_C;
    fsread["F"]>>m_F;
    fsread["label_string"]>>m_label_string;

    fsread.release();

    m_subModels.resize(m_n_models);
    for(int m=0;m<m_n_models;m++)
        m_subModels[m].load(m_modelPath+"subModel"+std::to_string(m)+".xml");
}

void ELM_IN_ELM_Model::query(const cv::Mat &mat, std::string &label)
{
    cv::Mat H(cv::Size(m_n_models*m_C,1),CV_32F);
    
    for(int m=0;m<m_n_models;m++)
    {
        cv::Mat ROI = H(cv::Range(0,1),cv::Range(m*m_C,(m+1)*m_C));
        m_subModels[m].query(mat,ROI);
        normalize(ROI);
    }
    
    cv::Mat output = H * m_F;
    std::cout<<output<<std::endl;
    int maxId = getMaxId(output);
    
    label.assign(m_label_string[maxId]);
}
