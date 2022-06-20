from ModelTraining.Preprocessing.DataPreprocessing.filters import ButterworthFilter, Filter

if __name__ == "__main__":
    filter_1 = ButterworthFilter(T=20, order=3)
    path = "../results"
    filename = "LPF.pickle"
    filter_1.save_pkl(path, filename)

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    print(filter_2.T)
    print(filter_2.order)
    print(filter_2.coef_)

    filter_3 = Filter.load_pkl(path, filename)
    print(filter_3.T)
    print(filter_3.order)
    print(filter_3.coef_)
