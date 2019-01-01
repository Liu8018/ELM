#ifndef DELM_H
#define DELM_H

#include "elm_model.h"

class DELM_Model
{
public:
    DELM_Model(int n_hiddenLayer, std::vector<int> hiddenNodes);
    
    void loadStandardDataset(const std::string datasetPath, const float trainSampleRatio,
                             const int resizeWidth, const int resizeHeight, 
                             const int channels, bool shuffle=true);
    
    void loadMnistData(const std::string path, const float trainSampleRatio, bool shuffle=true);
    
    void fit();
    
    float validate();
    
private:
    int m_n_hiddenLayer;
    std::vector<int> m_hiddenNodes;
    
    int m_width;
    int m_height;
    int m_channels;
    
    std::vector<std::string> m_label_string;
    
    std::vector<cv::Mat> m_trainImgs;
    std::vector<cv::Mat> m_testImgs;
    std::vector<std::vector<bool>> m_trainLabelBins;
    std::vector<std::vector<bool>> m_testLabelBins;
};

#endif // DELM_H
