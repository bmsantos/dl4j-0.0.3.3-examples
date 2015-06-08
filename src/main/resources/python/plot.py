import sys
from PIL import Image
from matplotlib.pyplot import hist, title, subplot, scatter
import matplotlib.pyplot as plot
from numpy import tanh, fabs, mean, ones, loadtxt, fromfile, zeros, product


def load_file(file_name):
    return loadtxt(file_name, delimiter=',')


def render_hbias(path):
    hMean = load_file(path)
    image = Image.fromarray(hMean * 256).show()


def render_hist_matrix(values, show=True, chart_title=''):
    if product(values.shape) < 2:
        values = zeros((3,3))
        chart_title += '-fake'
    hist(values)
    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)


def plot_multiple_matrices(data):
    paths = data.split(',')
    graph_count = 1
    print paths
    for idx, path in enumrate(paths):
        if idx % 2 == 0:
            print 'Loading matrix: ' + paths[path+idx] + '\n'
            matrix = load_file(path)
            subplot(2, len(paths)/3, graph_count)
            plot.tight_layout()
            render_hist_matrix(matrix, chart_title=paths[path+idx])
    plot.show()


def render_scatter_matrix(values, show=True, chart_title=''):
    if product(values.shape) < 2:
        values = zeros((3,3))
        chart_title += '-fake'
    scatter(values)
    magnitude = ' mm %g ' % mean(fabs(values))
    chart_title += ' ' + magnitude
    title(chart_title)


def plot_multiple_scatters(data):
    paths = data.split(',')
    graph_count = 1
    print paths
    for i  in xrange(len(paths) - 1):
        if i % 2 == 0:
            path = paths[i]
            title = paths[i + 1]
            print 'Loading matrix ' + path + '\n'
            matrix = load_file(path)
            subplot(2,len(paths) / 3,graph_count)
            plot.tight_layout()
            render_scatter_matrix(matrix, False, chart_title=title)
            graph_count += 1
    plot.show()

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
