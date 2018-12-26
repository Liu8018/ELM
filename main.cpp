#include "elm_in_elm_model.h"

int main()
{
    /*cv::Mat mat(6,6,CV_32F,cv::Scalar(1));
    mat.at<float>(0,4) = 3;
    mat.at<float>(2,3) = 4;
    mat.at<float>(4,2) = 5;
    mat.at<float>(5,1) = 6;
    mat.at<float>(1,2) = 7;
    mat.at<float>(3,3) = 4;
    mat.at<float>(2,4) = 2;
    mat.at<float>(5,5) = 1;
    
    std::cout<<"src:\n"<<mat<<std::endl;
    
    ELM_CNN_Model model;
    model.test(mat,mat);
    
    std::cout<<"dst:\n"<<mat<<std::endl;
    */
/*
    //加载ELM_IN_ELM模型
    ELM_IN_ELM_Model model(100,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
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
    std::cout<<runtime<<"s"<<std::endl;
*/
    

    //训练ELM_IN_ELM模型
    /*ELM_IN_ELM_Model model(80,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadStandardDataset("/run/media/liu/D/linux-windows/dataset/口袋妖怪", 0.6,30,30,3,true,false);
    for(int i=0;i<80;i++)
        model.setSubModelHiddenNodes(i,120+5*i);
    model.fitSubModels();
    model.fitMainModel();
    model.save();*/
    
    ELM_IN_ELM_Model model(1,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.5);
    //for(int i=0;i<10;i++)
        model.setSubModelHiddenNodes(0,5000);
    model.fitSubModels(30000);

    
    return 0;
}
