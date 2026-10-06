"""Vẽ đồ thị tính toán của một neuron (cần cài graphviz: pip install graphviz
và chương trình Graphviz trên máy).

Chạy từ thư mục gốc dự án:  python examples/visualize_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigrad import Value
from minigrad.viz import draw_dot

x1, x2 = Value(2.0, label='x1'), Value(0.0, label='x2')
w1, w2 = Value(-3.0, label='w1'), Value(1.0, label='w2')
b = Value(6.8814, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
n = x1w1 + x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()

dot = draw_dot(o)
dot.render('neuron_graph', cleanup=True)
print('Đã tạo neuron_graph.svg')
