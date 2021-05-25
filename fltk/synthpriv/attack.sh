# python -m fltk.synthpriv.attack purchase models/PurchaseMLP_DistPurchaseDataset_client1_50_start.model
# python -m fltk.synthpriv.attack purchase models/PurchaseMLP_DistPurchaseDataset_client1_50_start.model --loss
# python -m fltk.synthpriv.attack purchase models/PurchaseMLP_DistPurchaseDataset_client1_50_start.model --loss -l 15
# python -m fltk.synthpriv.attack purchase models/PurchaseMLP_DistPurchaseDataset_client1_50_start.model --loss -l 15 -g 18
# python -m fltk.synthpriv.attack purchase models/PurchaseMLP_DistPurchaseDataset_client1_50_start.model --loss -l 15 11 -g 18 14

# python -m fltk.synthpriv.attack adult models/AdultMLP_DistAdultDataset_client1_100_start.model
# python -m fltk.synthpriv.attack adult models/AdultMLP_DistAdultDataset_client1_100_start.model --loss
# python -m fltk.synthpriv.attack adult models/AdultMLP_DistAdultDataset_client1_100_start.model --loss -l 11
# python -m fltk.synthpriv.attack adult models/AdultMLP_DistAdultDataset_client1_100_start.model --loss -l 11 -g 14
# python -m fltk.synthpriv.attack adult models/AdultMLP_DistAdultDataset_client1_100_start.model --loss -l 11 7 -g 14 10

# python -m fltk.synthpriv.attack texas models/TexasMLP_DistTexasDataset_client1_100_start.model
# python -m fltk.synthpriv.attack texas models/TexasMLP_DistTexasDataset_client1_100_start.model --loss 
# python -m fltk.synthpriv.attack texas models/TexasMLP_DistTexasDataset_client1_100_start.model --loss -l 15 
# python -m fltk.synthpriv.attack texas models/TexasMLP_DistTexasDataset_client1_100_start.model --loss -l 15 -g 18
# python -m fltk.synthpriv.attack texas models/TexasMLP_DistTexasDataset_client1_100_start.model --loss -l 15 11 -g 18 14

python -m fltk.synthpriv.attack cifar models/DenseNet_DistCIFAR100Dataset_client1_100_start.model --loss 
python -m fltk.synthpriv.attack cifar models/AlexNet_DistCIFAR100Dataset_client1_100_start.model --loss 
python -m fltk.synthpriv.attack cifar models/Cifar100ResNet_DistCIFAR100Dataset_client1_100_start.model --loss 
