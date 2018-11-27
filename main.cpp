#include "elm_model.h"

int main()
{
    ELM_Model model;
    
    //model.setHiddenNodes(1500);
    model.loadStandardDataset("/run/media/liu/D/linux-windows/dataset/digit",0.2,16,16,1);
    model.fit();
    model.save("model.xml");
    
    
    /*model.load("model.xml");
    cv::Mat testImg = cv::imread("/run/media/liu/D/linux-windows/dataset/digit_test/3_2.png",0);
    std::string label;
    model.query(testImg,label);
    std::cout<<"result:"<<label<<std::endl;*/
    
    return 0;
}
