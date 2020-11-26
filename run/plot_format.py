from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


def formatnum(x, pos):
    return '$10^{}$'.format(int(x))


def plot_norm(ax, *args, title='', grid=False, formatter_x=False, formatter_y=False, formatter_z=False,
              x_lim=[], y_lim=[], z_lim=[], legend=True, legend_loc='upper left'):
    formatter1 = FuncFormatter(formatnum)
    if formatter_x:
        ax.xaxis.set_major_formatter(formatter1)
    if formatter_y:
        ax.yaxis.set_major_formatter(formatter1)
    # if formatter_z:
    #     ax.zaxis.set_major_formatter(formatter1)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font_legend = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 14}
    font_label = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 15}
    font_title = {'family': 'DejaVu Sans', 'weight': 'bold', 'size': 18}

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    # if z_lim:
    #     ax.set_zlim(z_lim[0], z_lim[1])
    if legend:
        ax.legend(loc=legend_loc, prop=font_legend)
    if grid:
        ax.grid(ls='-.')
    ax.set_xlabel(args[0], font_label)
    ax.set_ylabel(args[1], font_label)
    # if len(args) == 3:
    #     if formatter_z:
    #         ax.zaxis.set_major_formatter(formatter1)
    #     if z_lim:
    #         ax.set_zlim(z_lim[0], z_lim[1])
    # ax.set_zlabel(args[2], font_label)
    ax.set_title(title, font_title)
    plt.tight_layout()