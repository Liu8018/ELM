#include "delm_model.h"

int main()
{
    //训练ELM_IN_ELM模型
    /*ELM_IN_ELM_Model model(80,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadStandardDataset("/run/media/liu/D/linux-windows/dataset/口袋妖怪", 0.6,30,30,3,true,false);
    for(int i=0;i<80;i++)
        model.setSubModelHiddenNodes(i,120+5*i);
    model.fitSubModels();
    model.fitMainModel();
    model.save();*/
    
    
    /*int n_model = 512;
    ELM_IN_ELM_Model model(n_model,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.5);
    for(int n=0;n<n_model;n++)
        model.setSubModelHiddenNodes(n,8);
    model.fitSubModels();
    model.fitMainModel();*/
    //model.save();
    
    
    //ELM_IN_ELM_Model model2;
    //model2.load("/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    //model2.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_test.csv",0);
    //model2.validate();
    
    //贪婪法训练
    /*int n_model = 5;
    ELM_IN_ELM_Model model(n_model,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.3);
    for(int n=0;n<n_model;n++)
        model.setSubModelHiddenNodes(n,80);
    model.init_greedyFitWhole(3);*/
    
    /*ELM_Model model;
    model.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.3);
    model.setHiddenNodes(200);
    model.fit();*/
    
    int n_hiddenLayer = 3;
    std::vector<int> hiddenNodes(n_hiddenLayer,20);
    //hiddenNodes.push_back(1024);
    //hiddenNodes.push_back(512);
    //hiddenNodes.push_back(256);
    //hiddenNodes.push_back(128);
    //hiddenNodes.push_back(64);
    
    DELM_Model model(n_hiddenLayer,hiddenNodes);
    model.loadMnistData("/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.3);
    model.fit();
    
    return 0;
}
