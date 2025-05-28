import os
import pickle

from data_utils.load_bonn_data import load_bonn_dataset
from data_utils.load_brainlat_data import load_brainlat_dataset
from data_utils.load_Alzheimer_data import load_Alzheimer_dataset
from data_utils.load_autism_data import load_autism_dataset
from data_utils.load_Epilepsy_data import load_Epilepsy_dataset
from data_utils.load_MDD_data import load_MDD_dataset
from data_utils.load_DREAMER_data import load_DREAMER_dataset



def load_VGL_dataset(args):
    VGL_dataset_pickle_path = os.path.join(args.data_dir, args.dataset, "VGL_dataset.pickle")

    if os.path.exists(VGL_dataset_pickle_path):
        print("load existing VGL dataset")
        with open(VGL_dataset_pickle_path, "rb") as f:
            VGL_train_data, VGL_test_data = pickle.load(f)
    else:
        print("construct VGL dataset")
        if args.dataset == "bonn":
            VGL_train_data, VGL_test_data = load_bonn_dataset(args)
        elif args.dataset =="Epilepsy":
            VGL_train_data, VGL_test_data = load_Epilepsy_dataset(args)
        elif args.dataset =="Alzheimer":
            VGL_train_data, VGL_test_data = load_Alzheimer_dataset(args)
        elif args.dataset =="autism":
            VGL_train_data, VGL_test_data = load_autism_dataset(args)
        elif args.dataset =="MDD":
            VGL_train_data, VGL_test_data = load_MDD_dataset(args)
            pass
        elif args.dataset =="DREAMER":
            VGL_train_data, VGL_test_data = load_DREAMER_dataset(args)
            pass
        elif args.dataset == "brainlat":
            VGL_train_data, VGL_test_data = load_brainlat_dataset(args)
        else:
            pass

        with open(VGL_dataset_pickle_path, "wb") as f:
            pickle.dump((VGL_train_data, VGL_test_data), f)
            print(f"VGL dataset has been saved to {VGL_dataset_pickle_path}")


    return VGL_train_data, VGL_test_data