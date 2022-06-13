# from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


def formatnum(x, pos):
    return '$10^{}$'.format(int(x))


def plot_norm(ax, xlabel=None, ylabel=None, zlabel=None, title=None, x_lim=[], y_lim=[], z_lim=[], legend=True,
              grid=False, frameon=True, legend_loc='upper left', font_color='black', legendsize=11, labelsize=14,
              titlesize=15, ticksize=13, linewidth=2, fontname='Arial', legendWeight='normal', labelWeight='bold',
              titleWeight='bold'):
    """
    画图模板
    :param ax: 轴
    :param xlabel: 横坐标
    :param ylabel: 纵坐标
    :param zlabel: 三维坐标
    :param title: 标题
    :param x_lim: 横轴范围
    :param y_lim: 纵轴范围
    :param z_lim: 三维范围
    :param legend: 是否显示图例
    :param grid: 是否显示网格
    :param frameon: 是否显示图例背景
    :param legend_loc: 图例位置
    :param font_color: 字体颜色
    :param legendsize: 图例字号
    :param labelsize: 标签字号
    :param titlesize: 标题字号
    :param ticksize: 坐标轴字号
    :param linewidth: 坐标轴宽度
    :param fontname: 字体类型
    :param legendWeight: 图例字体粗细
    :param labelWeight: 标签字体粗细
    :param titleWeight: 标题字体粗细
    :return:
    """
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)

    # 设置坐标刻度值的大小以及刻度值的字体 Arial, Times New Roman
    ax.tick_params(which='both', width=linewidth, labelsize=ticksize, colors=font_color)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(fontname) for label in labels]

    font_legend = {'family': fontname, 'weight': legendWeight, 'size': legendsize}
    font_label = {'family': fontname, 'weight': labelWeight, 'size': labelsize, 'color': font_color}
    font_title = {'family': fontname, 'weight': titleWeight, 'size': titlesize, 'color': font_color}

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    if z_lim:
        ax.set_zlim(z_lim[0], z_lim[1])
    if legend:
        ax.legend(loc=legend_loc, prop=font_legend, frameon=frameon)
    if grid:
        ax.grid(ls='-.')
    if xlabel:
        ax.set_xlabel(xlabel, font_label)
    if ylabel:
        ax.set_ylabel(ylabel, font_label)
    if zlabel:
        ax.set_zlabel(zlabel, font_label)
    if title:
        ax.set_title(title, font_title)
    plt.tight_layout()
