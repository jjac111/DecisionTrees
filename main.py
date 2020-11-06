from utils import *

prepare_data()

make_folded_sets()

metrics = classify('CART')

metrics