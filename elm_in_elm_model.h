#ifndef ELM_IN_ELM_MODEL_H
#define ELM_IN_ELM_MODEL_H

#include "elm_model.h"

class ELM_IN_ELM_Model
{
public:
    ELM_IN_ELM_Model(const int n_models,const std::string modelDir);
    
    void setSubModelHiddenNodes(const int modelId, const int n_nodes);
    
    void loadStandardDataset(const std::string path, const float trainSampleRatio,
                             const int resizeWidth, const int resizeHeight, 
                             const int channels, bool validate=true,bool shuffle=true);
    
    void loadMnistData(const std::string path, const float trainSampleRatio, bool validate=true, bool shuffle=true);
    
    void fitSubModels(int batchSize = -1);
    void fitMainModel(const int Q=-1);
    
    void save();
    void load();
    
    void query(const cv::Mat &mat, std::string &label);
    
private:
    int m_n_models;
    std::vector<int> m_subModelHiddenNodes;
    
    bool m_validate;
    
    ELM_Model m_subModelToTrain;
    
    bool m_shuffle;
    
    std::vector<ELM_Model> m_subModels;
    
    std::string m_datasetPath;
    std::string m_modelPath;
    int m_width;
    int m_height;
    float m_trainSampleRatio;
    int m_channels;
    int m_Q;
    int m_C;
    
    cv::Mat m_F;
    
    std::vector<std::string> m_label_string;
    
};

#endif // ELM_IN_ELM_MODEL_H
