from sklearn.tree import export_graphviz
from ...datamodels.datamodels.randomforest import RandomForestRegression
import pydotplus


def visualize_rf(ml_model: RandomForestRegression, feature_names=None, outfile_name="rf", depth=5, max_estimators = 1):
    """
    Visualize random forest
    @param ml_model: RandomForest model
    @param feature_names: List of input feature names
    """
    estimators = ml_model.model.estimators_[0:max_estimators]
    for index, estimator in enumerate(estimators):
        dot_data = export_dot_file(estimator, f"{outfile_name}_{index}", feature_names, depth)
        export_graph_from_dot_data(dot_data, f"{outfile_name}_{index}")


def export_dot_file(estimator, filename="./rf", feature_names=None, depth=5):
    """
    Export dot file
    @param estimator: sklearn rf estimator
    @param filename: output filename
    @param feature_names: list of feature names
    @param depth: depth of tree to output
    @return: dot_data structure
    """
    export_graphviz(estimator, out_file=f"{filename}.dot", feature_names=feature_names, max_depth=depth)
    return export_graphviz(estimator, out_file=None, feature_names=feature_names, max_depth=depth)


def export_graph_from_dot_data(dot_data, filename="./rf"):
    """
    Export graph
    @param dot_data: dot data
    @param filename: output filename
    """
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png(f"{filename}.png")

