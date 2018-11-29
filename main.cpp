#include "elm_in_elm_model.h"

int main()
{
    /*ELM_IN_ELM_Model model(100,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.load();
    
    cv::Mat src = cv::imread("/home/liu/下载/cnn-keras/examples/squirtle_plush.png",0);
    std::string label;
    float runtime = cv::getTickCount();
    model.query(src,label);
    runtime = (cv::getTickCount()-runtime)/cv::getTickFrequency();
    std::cout<<label<<std::endl;
    std::cout<<runtime<<"s"<<std::endl;
    
    src = cv::imread("/home/liu/下载/cnn-keras/examples/pikachu_toy.png",0);
    runtime = cv::getTickCount();
    model.query(src,label);
    runtime = (cv::getTickCount()-runtime)/cv::getTickFrequency();
    std::cout<<label<<std::endl;
    std::cout<<runtime<<"s"<<std::endl;*/
    
    ELM_IN_ELM_Model model(50,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadStandardDataset("/run/media/liu/D/linux-windows/dataset/digit",0.1,16,16,1,true,true);
    model.fitSubModels();
    model.fitMainModel();
    model.save();
    
    /*ELM_Model model;
    
    model.setHiddenNodes(200);
    model.loadStandardDataset("/run/media/liu/D/linux-windows/dataset/digit",0.2,16,16,1);
    model.fit();
    model.save("model.xml");*/
    
    
    /*model.load("model.xml");
    cv::Mat testImg = cv::imread("/run/media/liu/D/linux-windows/dataset/digit_test/3_2.png",0);
    std::string label;
    model.query(testImg,label);
    std::cout<<"result:"<<label<<std::endl;*/
    
    return 0;
}
