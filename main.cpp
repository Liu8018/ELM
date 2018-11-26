#include "elm_model.h"

int main()
{
    ELM_Model model;
    
    //model.setHiddenNodes(80);
    model.loadStandardDataset("/home/liu/下载/cnn-keras/dataset",0.2,50,50,3);
    model.fit();
    
    return 0;
}
