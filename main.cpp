#include "elm_in_elm_model.h"

int main()
{
    //训练ELM_IN_ELM模型.一般数据集
    /*int nmodels=20;
    ELM_IN_ELM_Model model(nmodels,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadStandardDataset("/media/liu/D/linux-windows/dataset/50_10photos", 0.7,100,100,1);
    for(int i=0;i<nmodels;i++)
        model.setSubModelHiddenNodes(i,200);
    model.fitSubModels();
    model.fitMainModel();
    model.save();*/
    
    //训练ELM_IN_ELM模型.MNIST数据集
    int n_model = 10;
    ELM_IN_ELM_Model model(n_model,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadMnistData("/media/liu/D/linux-windows/dataset/MNIST_data2/debug.csv",1);
    for(int n=0;n<n_model;n++)
        model.setSubModelHiddenNodes(n,20);
    model.fitSubModels();
    model.fitMainModel();
    model.save();
    
    //载入ELM_IN_ELM模型,再次训练
    /*ELM_IN_ELM_Model model2;
    model2.load("/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model2.loadMnistData("/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_test.csv",0.5);
    model2.fitSubModels();
    model2.fitMainModel();
    model2.save();*/
    //model2.validate();
    
    //贪婪法训练
    /*int n_model = 5;
    ELM_IN_ELM_Model model(n_model,"/home/liu/codes/项目/ELM/trained_ELM_IN_ELM_models/a/");
    model.loadMnistData("/run/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.3);
    for(int n=0;n<n_model;n++)
        model.setSubModelHiddenNodes(n,80);
    model.init_greedyFitWhole(3);*/
    
    /*ELM_Model model;
    //model.loadMnistData("/media/liu/D/linux-windows/dataset/MNIST_data2/mnist_train.csv",0.3);
    model.loadStandardDataset("/media/liu/D/linux-windows/dataset/20_10photos",0.7,50,50,1);
    model.setHiddenNodes(1200);
    model.fit();*/
    
    return 0;
}
