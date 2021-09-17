from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize


def voronoi(points, save=False):
    """
    生成德劳内三角网和泰森多边形
    :param points: 要生成德劳内三角网和泰森多边形的点
    :param save: 默认不保存本地文件
    :return:
    """
    pointlength = len(points)
    array = np.array(points)

    # 泰森多边形
    vor = Voronoi(array, furthest_site=False, incremental=True, qhull_options=None)

    if save:
        # 泰森多边形的顶点
        vertices = vor.vertices
        with open('voronoi_vertices.txt', 'w', encoding='utf-8') as f:
            for index, v in enumerate(vertices):
                f.write(str(index) + '\t' + 'POINT(' + str(v[0]) + ' ' + str(v[1]) + ')' + '\n')

        # 泰森多边形的面，-1代表无穷索引
        regions = vor.regions
        with open('voronoi_regions.txt', 'w', encoding='utf-8') as f:
            for index, r in enumerate(regions):
                if len(r) == 0:
                    continue
                if -1 in r:
                    continue
                angulars = []
                for id in r:
                    angulars.append(vertices[id])
                angulars.append(vertices[r[0]])
                polygon = Polygon(angulars)
                f.write(str(index) + '\t' + str(polygon.wkt) + '\n')

        # 德劳内三角形的边，用原始的点数量
        vorOriginal = Voronoi(array[0:pointlength], furthest_site=False, incremental=True, qhull_options=None)
        ridge_points = vorOriginal.ridge_points
        polylines = []
        for ridge in ridge_points:
            polyline = LineString([points[ridge[0]], points[ridge[1]]])
            polylines.append(polyline)
        # 德劳内三角形构面
        delaunays = polygonize(polylines)
        with open(r'voronoi_delaunays.txt', 'w', encoding='utf-8') as f:
            for index, p in enumerate(delaunays):
                f.write(str(index) + '\t' + str(p.wkt) + '\n')

    fig = voronoi_plot_2d(vor)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.show()


def data_position(allAtoms, dim, atomsPerUnitCell, cellNums, latticeConstant, orient=None):
    """
    Atomic structure model
    :param allAtoms: the total number of atoms
    :param dim: dimensions of simulation
    :param atomsPerUnitCell: atoms numbers per unit cell
    :param cellNums: cell numbers
    :param latticeConstant: lattice constant matrix
    :param orient: orientation of cell
    :return:
    """
    # Define the normalized atomic coordinates of FCC crystal structure
    if orient is None:
        orient = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        normalizedCoordinates = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]) * latticeConstant
    elif orient == [[1, 1, -2], [-1, 1, 0], [1, 1, 1]]:
        normalizedCoordinates = np.array([[0, 0, 0], [0.66666667, 0.66666667, 0], [0.33333333, 0.33333333, 0],
                                          [0.5, 0, 0.5], [0.83333333, 0.33333333, 0.5], [0.16666667, 0.66666667, 0.5]])\
                                * latticeConstant

    position = np.zeros([allAtoms, dim])
    n = 0
    if dim == 3:
        for i in range(cellNums[0]):
            for j in range(cellNums[1]):
                for k in range(cellNums[2]):
                    for l in range(atomsPerUnitCell):
                        position[n, :] = latticeConstant * np.array([i, j, k]) + normalizedCoordinates[l, :]
                        n += 1

    return position


def rotatePosition(position, center, angle):
    position -= center
    m = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    position = np.dot(m, position.T)
    return position.T + center


def convertToLammps(position, file, allAtoms, cellNums, latticeConstant):
    """
    Convert atomic structure model to LAMMPS data format
    :param position:
    :param file:
    :param allAtoms: the total number of atoms
    :param cellNums: cell numbers
    :param latticeConstant: lattice constant matrix
    :return:
    """
    with open(file, 'w') as f:
        f.write('# LAMMPS timestep\t0\n\n')
        f.write('{}\tatoms\n'.format(allAtoms))
        f.write('{}\tatom types\n\n'.format(1))

        f.write('{}\t{}\txlo\txhi\n'.format(0.0, latticeConstant[0] * cellNums[0]))
        f.write('{}\t{}\tylo\tyhi\n'.format(0.0, latticeConstant[1] * cellNums[1]))
        f.write('{}\t{}\tzlo\tzhi\n\n'.format(0.0, latticeConstant[2] * cellNums[2]))

        f.write('Masses\n\n')
        f.write('{}\t{:.4f}\t# {}\n\n'.format(1, 58.6934, 'Ni'))

        f.write('Atoms # atomic\n\n')
        for i, pos in enumerate(position, 1):
            f.write('{}\t1\t{}\t{}\t{}\n'.format(i, *pos))


if __name__ == '__main__':
    dim = 3
    atomsPerUnitCell = 6
    cellNums = [10, 8, 15]
    latticeConstant = [4.31600093, 6.10374705, 2.49184430]
    position = data_position(6 * 10 * 8 * 15, dim, atomsPerUnitCell, cellNums, latticeConstant,
                             [[1, 1, -2], [-1, 1, 0], [1, 1, 1]])
    file = r'C:\\Users\\Yuan\\Desktop\\test.lmp'
    convertToLammps(position, file + 'origin', 6 * 10 * 8 * 15, cellNums, latticeConstant)
    position = rotatePosition(position, [20, 20, 15], pi / 2)
    convertToLammps(position, file, 6 * 10 * 8 * 15, cellNums, latticeConstant)