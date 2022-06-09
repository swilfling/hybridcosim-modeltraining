from ModelTraining.Preprocessing.DataPreprocessing import ButterworthFilter

if __name__ == "__main__":
    filter = ButterworthFilter(T=20, order=3)
    path = "../results"
    filename = "LPF.pickle"
    filter.save_pkl(path, filename)

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    print(filter_2.T)
    print(filter_2.order)
    print(filter_2.coef_)