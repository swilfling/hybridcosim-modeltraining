from ModelTraining.Preprocessing.DataPreprocessing.Filter import ButterworthFilter

if __name__ == "__main__":
    filter = ButterworthFilter(T=20, order=3)
    path = "../results"
    filename = "LPF.pickle"
    filter.save(path, filename)

    filter_2 = ButterworthFilter.load(path, filename)
    print(filter_2.T)
    print(filter_2.order)
    print(filter_2.coeffs)