import os
from sodistinct.io.loader import load_graph

def test_load_edgelist(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("0 1\n1 2\n")

    g = load_graph(str(p))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
