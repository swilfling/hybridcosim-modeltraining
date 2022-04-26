from sklearn.tree import export_graphviz
import pydotplus


def visualize_rf(ml_model, feature_names, outfile_name="rf", depth=5, max_estimators = 1):
    estimators = ml_model.model.estimators_[0:max_estimators]
    for index, estimator in enumerate(estimators):
        dot_data = export_dot_file(estimator, f"{outfile_name}_{index}", feature_names, depth)
        export_graph_from_dot_data(dot_data, f"{outfile_name}_{index}")


def export_dot_file(estimator, filename, feature_names, depth):
    export_graphviz(estimator, out_file=f"{filename}.dot", feature_names=feature_names, max_depth=depth)
    return export_graphviz(estimator, out_file=None, feature_names=feature_names, max_depth=depth)


def export_graph_from_dot_data(dot_data, filename):
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png(f"{filename}.png")

