from visualize import visualize_pcd
from data_utils import read_ply

src1 = read_ply("../test_data/src1.ply")
trg1 = read_ply("../test_data/trg1.ply")

visualize_pcd(src1, trg1)