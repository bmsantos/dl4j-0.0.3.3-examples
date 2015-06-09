'''
Optimization Methods Visualalization

Graph tools to help visualize how optimization is performing
'''

from matplotlib.pyplot import hist, title, subplot, scatter
import matplotlib.pyplot as plt
from numpy import tanh, fabs, mean, ones, loadtxt, fromfile, zeros, product
import seaborn
from PIL import Image
import sys


def load_file(file_name):
    return loadtxt(file_name, delimiter=',')


def render_hbias(path):
    hMean = load_file(path)
    image = Image.fromarray(hMean * 256).show()
    return

def render_hist_matrix(values, chart_title=''):
    if product(values.shape) < 2:
        values = zeros((3, 3))
        chart_title += '-fake'

    hist(values)
    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)
    return

def plot_multiple_matrices(data):
    paths = data.split(',')

    for idx, path in enumerate(paths):
        if idx % 2 == 0:
            print 'Loading matrix: ' + paths[idx+1] + '\n'
            matrix = load_file(paths[idx])
            subplot(2, len(paths)/4, idx/2+1) #subplot index starts at 1
            render_hist_matrix(matrix, chart_title=paths[idx+1])

    plt.tight_layout()
    plt.show()
    return

def render_scatter_matrix(values, chart_title=''):
    if product(values.shape) < 2:
        values = zeros((3, 3))
        chart_title += '-fake'

    scatter(values)
    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)


def plot_multiple_scatters(data):
    paths = data.split(',')

    for idx, path in enumerate(paths):
        if idx % 2 == 0:
            title = paths[idx + 1]
            print 'Loading matrix ' + path + '\n'
            matrix = load_file(path)
            subplot(2, len(paths)/4, idx/2+1)
            render_scatter_matrix(matrix, chart_title=title)

    plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Please specify a command: One of hbias,weights,plot and a file path'
        sys.exit(1)
    plot_type = sys.argv[1]
    data = sys.argv[2]
    if plot_type == 'hbias':
        render_hbias(data)
    elif plot_type == 'weights':
        render_hist_matrix(data)
    elif plot_type == 'multi':
        plot_multiple_matrices(data)
    elif plot_type == 'scatter':
        plot_multiple_scatters(data)
