from utils import *

prepare_data()

make_folded_sets()

metrics = classify('C4.5')
plot_results(metrics)