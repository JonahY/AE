from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize
from tqdm import tqdm
from scipy.linalg import solve


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


def dataPosition(allAtoms, dim, atomsPerUnitCell, cellNums, latticeConstant, orient=None):
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
    if orient == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        normalizedCoordinates = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]) * latticeConstant
    elif orient == [[1, 1, -2], [-1, 1, 0], [1, 1, 1]]:
        normalizedCoordinates = np.array([[0, 0, 0], [0.66666667, 0.66666667, 0], [0.33333333, 0.33333333, 0],
                                          [0.5, 0, 0.5], [0.83333333, 0.33333333, 0.5], [0.16666667, 0.66666667, 0.5]]) \
                                * latticeConstant

    position = np.zeros([allAtoms, dim])
    n = 0
    if dim == 3:
        for i in tqdm(range(cellNums[0])):
            for j in range(cellNums[1]):
                for k in range(cellNums[2]):
                    for l in range(atomsPerUnitCell):
                        position[n, :] = latticeConstant * np.array([i, j, k]) + normalizedCoordinates[l, :]
                        n += 1

    return position


def rotatePosition(position, center, angle):
    """
    Rotate atomic coordinates around the Z axis
    :param position:
    :param center:
    :param angle:
    :return:
    """
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
        for i, pos in tqdm(enumerate(position, 1)):
            f.write('{}\t1\t{}\t{}\t{}\n'.format(i, *pos))


def getOuterCircle(A, B, C):
    """
    Calculate the center of the circumcircle
    :param A:
    :param B:
    :param C:
    :return:
    """
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    xc, yc = C[0], C[1]

    # midpoint of two edges
    x1, y1 = (xa + xb) / 2.0, (ya + yb) / 2.0
    x2, y2 = (xb + xc) / 2.0, (yb + yc) / 2.0

    # slopes of the two lines
    ka = (yb - ya) / (xb - xa) if xb != xa else None
    kb = (yc - yb) / (xc - xb) if xc != xb else None

    alpha = np.arctan(ka) if ka != None else np.pi / 2
    beta = np.arctan(kb) if kb != None else np.pi / 2

    # slope of the two perpendicular bisectors
    k1 = np.tan(alpha + np.pi / 2)
    k2 = np.tan(beta + np.pi / 2)

    # centre
    y, x = solve([[1.0, -k1], [1.0, -k2]], [y1 - k1 * x1, y2 - k2 * x2])
    return x, y


def getFootOfPerpendicular(O, A, B):
    """
    Calculate the vertical foot from point to line
    :param O: point outside the line
    :param A: starting point of a line
    :param B: end point of line
    :return:
    """
    dx = B[0] - A[0]
    dy = B[1] - A[1]

    if not dx and not dy:
        return A

    k = ((O[0] - A[0]) * dx + (O[1] - A[1]) * dy) / (dx ** 2 + dy ** 2)
    xn = k * dx + A[0]
    yn = k * dy + A[1]
    return xn, yn


def getLinearEquation(A, B, origin=False):
    """
    根据两点坐标计算直线方程
    :param A: starting point of a line
    :param B: end point of line
    :param origin: Whether to transform the equation of the line to the initial coordinate system of the model
    :return:
    """
    dx = B[0] - A[0]
    dy = B[1] - A[1]

    if not dx:
        return 'x {} = 0'.format('- %f' % A[0] if A[0] >= 0 else '+ %f' % -A[0]), 1, 0, -A[0]
    elif not dy:
        return 'y {} = 0'.format('- %f' % A[1] if A[1] >= 0 else '+ %f' % -A[1]), 0, 1, -A[1]

    k = dy / dx
    if not origin:
        return '{} * Position.X + Position.Y {} = 0'.format(-k, '+ %f' % (k * A[0] - A[1]) if k * A[0] >= A[1]
                else k * A[0] - A[1]), -k, 1, k * A[0] - A[1]
    else:
        return '{} * (Position.X + 600) + Position.Y + 400 {} = 0'.format(-k, '+ %f' % (k * A[0] - A[1]) if k * A[0] >= A[1]
                else k * A[0] - A[1]), -k, 1, k * A[0] - A[1]


def getEquationResult(a, c, x):
    """
    Calculate y
    :param a:
    :param c:
    :param x:
    :return:
    """
    return -a * x - c


def getCrossPoint(line1, line2):
    """
    Calculate the coordinates of the intersection of two lines
    :param line1:
    :param line2:
    :return:
    """
    x1 = line1[0][0]  # 取四点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if not k2:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return x, y


if __name__ == '__main__':
    '''
    dim = 3
    # orient = [[1, 1, -2], [-1, 1, 0], [1, 1, 1]]
    orient = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    atomsPerUnitCell = 4
    cellNums = [170, 113, 1]
    allAtoms = atomsPerUnitCell * cellNums[0] * cellNums[1] * cellNums[2]
    # latticeConstant = [4.31600093, 6.10374705, 2.49184430]
    latticeConstant = [3.524, 3.524, 3.524]
    position = dataPosition(allAtoms, dim, atomsPerUnitCell, cellNums, latticeConstant, orient)
    file = r'C:\\Users\\Yuan\\Desktop\\test.lmp'
    convertToLammps(position, file + '.origin', allAtoms, cellNums, latticeConstant)
    position = rotatePosition(position, list(map(lambda i: i//2, cellNums)), pi / 4)
    convertToLammps(position, file, allAtoms, cellNums, latticeConstant)
    '''

    temp = 'node 287.762045 350.586382\n' \
           'node 31.244322 154.925875\n' \
           'node 86.687094 240.067821\n' \
           'node 449.481089 377.009174\n' \
           'node 517.696848 74.026584\n' \
           'node 503.496165 243.600690\n' \
           'node 181.791053 17.572401\n' \
           'node 571.052643 10.826122\n' \
           'node 79.652767 263.637397\n' \
           'node 597.346732 18.727028\n' \
           'node 295.776037 232.424923\n' \
           'node 31.367135 88.699926'
    points_origin = np.array([list(map(lambda j: float(j), i.split(' ')[1:])) for i in temp.split('\n')])
    points = np.append(points_origin, points_origin + [0, 400], axis=0)
    points = np.append(points, points_origin + [0, 800], axis=0)
    points = np.append(points, points_origin + [600, 0], axis=0)
    points = np.append(points, points_origin + [600, 400], axis=0)
    points = np.append(points, points_origin + [600, 800], axis=0)
    points = np.append(points, points_origin + [1200, 0], axis=0)
    points = np.append(points, points_origin + [1200, 400], axis=0)
    points = np.append(points, points_origin + [1200, 800], axis=0)
    centers = points_origin + [600, 400]

    with open(r'C:\Users\Yuan\Desktop\voronoi_vertices.txt', 'r') as f:
        vertices = [list(map(lambda j: float(j), i.strip().split('\t')[1][6:-1].split(' '))) for i in f.readlines()]
    vertices = np.array(vertices)
    vertices = vertices[
        np.where((vertices[:, 0] > 600) & (vertices[:, 0] < 1200) & (vertices[:, 1] > 400) & (vertices[:, 1] < 800))[0]]
    vertices = np.array(sorted(vertices, key=lambda i: (i[0], i[1])))

    # ----------------------------------------------- subdivision grain -----------------------------------------------
    voronoi(points)
    plt.ylim(300, 900)
    plt.xlim(400, 1400)
    plt.axhline(400, 0.2, 0.8, c='black')
    plt.axhline(800, 0.2, 0.8, c='black')
    plt.axvline(600, 1 / 6, 5 / 6, c='black')
    plt.axvline(1200, 1 / 6, 5 / 6, c='black')

    # ---------------------------------------------------- grain 1 ----------------------------------------------------
    abc = np.array([[825, 710], [830, 740], [920, 705], [870, 790], [920, 865], [940, 750]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0], abc[:, 1] - 400, s=15, marker='x', c='black')
    print('-' * 20 + 'grain 1' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [6, 9, 13], [10, 13, '1']):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == 10:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], vertices[j] + [0, 400]])
        elif j == '1':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal, yCal - 400, c='b')

    for abc_A, abc_B, i, j in zip([3, 3, 4], [4, 5, 5], [6, 13, 11], [10, '1', 13]):
        x0, y0 = getOuterCircle(abc[3], abc[4], abc[5])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == 10:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], vertices[j] + [0, 400]])
        elif j == '1':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] + [0, 400], vertices[j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal, yCal - 400, c='b')

    xc, yc = getCrossPoint([vertices[13], centers[0]], [vertices[6], vertices[10] + [0, 400]])
    equation, a, b, c = getLinearEquation(vertices[13], [xc, yc])
    print(equation)
    xCal = np.linspace(vertices[13][0], xc, 50)
    yCal = getEquationResult(a, c, xCal)
    plt.plot(xCal, yCal, c='b')
    plt.plot(xCal, yCal - 400, c='b')

    # ---------------------------------------------------- grain 2 ----------------------------------------------------
    abc = np.array([[590, 550], [630, 595], [670, 545]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0] + 600, abc[:, 1], s=15, marker='x', c='black')
    print('-' * 20 + 'grain 2' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [18, 19, 0], [23, 4, 5]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == 23:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]] - [600, 0])
        elif j == 4:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] - [600, 0], vertices[j]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal + 600, yCal, c='b')

    # ---------------------------------------------------- grain 3 ----------------------------------------------------
    abc = np.array([[670, 610], [765, 650], [760, 560]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    print('-'*20 + 'grain 3' + '-'*20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [0, 0, 7], [8, 5, 8]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')

    # ---------------------------------------------------- grain 4 ----------------------------------------------------
    abc = np.array([[980, 870], [1000, 725], [1080, 770]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0], abc[:, 1] - 400, s=15, marker='x', c='black')
    print('-' * 20 + 'grain 4' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [11, 12, 14], [13, 16, 17]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == 13:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] + [0, 400], vertices[j]])
        elif j == 16:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]] + [0, 400])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal, yCal - 400, c='b')

    # ---------------------------------------------------- grain 5 ----------------------------------------------------
    abc = np.array([[1030, 520], [1110, 525], [1100, 460]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    print('-' * 20 + 'grain 5' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [15, 12, 19], [18, 16, 21]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')

    # ---------------------------------------------------- grain 6 ----------------------------------------------------
    abc = np.array([[1060, 685], [1130, 710], [1140, 670], [1050, 635], [1060, 590], [1140, 620]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    print('-' * 20 + 'grain 6' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [14, 14, 22], [17, 23, 23]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')

    for abc_A, abc_B, i, j in zip([3, 3, 4], [4, 5, 5], [14, 14, 15], [15, 23, 18]):
        x0, y0 = getOuterCircle(abc[3], abc[4], abc[5])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')

    equation, a, b, c = getLinearEquation(vertices[14], vertices[23])
    print(equation)
    xCal = np.linspace(vertices[14][0], vertices[23][0], 50)
    yCal = getEquationResult(a, c, xCal)
    plt.plot(xCal, yCal, c='b')

    # ---------------------------------------------------- grain 7 ----------------------------------------------------
    abc = np.array([[740, 450], [770, 520], [840, 470], [715, 390], [755, 340], [810, 405]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0], abc[:, 1] + 400, s=15, marker='x', c='black')
    print('-' * 20 + 'grain 7' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [3, 3, 7], [4, '7', 10]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == '7':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal, yCal + 400, c='b')

    for abc_A, abc_B, i, j in zip([3, 3, 4], [4, 5, 5], [2, 3, 6], [6, '7', 10]):
        x0, y0 = getOuterCircle(abc[3], abc[4], abc[5])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == '7':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        elif j == 6:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]] - [0, 400])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] - [0, 400], vertices[j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal, yCal + 400, c='b')

    xc, yc = getCrossPoint([vertices[3], centers[6]], [vertices[6] - [0, 400], vertices[10]])
    equation, a, b, c = getLinearEquation(vertices[3], [xc, yc])
    print(equation)
    xCal = np.linspace(vertices[3][0], xc, 50)
    yCal = getEquationResult(a, c, xCal)
    plt.plot(xCal, yCal, c='b')
    plt.plot(xCal, yCal + 400, c='b')

    # ---------------------------------------------------- grain 9 ----------------------------------------------------
    abc = np.array([[610, 700], [650, 725], [645, 670], [710, 725], [705, 675], [755, 695]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0] + 600, abc[:, 1], s=15, marker='x', c='black')
    print('-' * 20 + 'grain 9' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [1, 22, 2], [2, 23, '9']):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if i == 22:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]] - [600, 0])
        elif j == '9':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal + 600, yCal, c='b')

    for abc_A, abc_B, i, j in zip([3, 3, 4], [4, 5, 5], [2, 2, 0], ['9', 6, 8]):
        x0, y0 = getOuterCircle(abc[3], abc[4], abc[5])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if j == '9':
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i], centers[int(j) - 1]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal + 600, yCal, c='b')

    xc, yc = getCrossPoint([vertices[2], centers[8]], vertices[[0, 8]])
    equation, a, b, c = getLinearEquation(vertices[2], [xc, yc])
    print(equation)
    xCal = np.linspace(vertices[2][0], xc, 50)
    yCal = getEquationResult(a, c, xCal)
    plt.plot(xCal, yCal, c='b')
    plt.plot(xCal + 600, yCal, c='b')

    # ---------------------------------------------------- grain 12 ----------------------------------------------------
    abc = np.array([[610, 490], [695, 500], [680, 450]])
    plt.scatter(abc[:, 0], abc[:, 1], s=15, marker='x', c='black')
    plt.scatter(abc[:, 0] + 600, abc[:, 1], s=15, marker='x', c='black')
    print('-' * 20 + 'grain 12' + '-' * 20)
    for abc_A, abc_B, i, j in zip([0, 0, 1], [1, 2, 2], [19, 21, 3], [4, 3, 4]):
        x0, y0 = getOuterCircle(abc[0], abc[1], abc[2])
        xn, yn = getFootOfPerpendicular([x0, y0], abc[abc_A], abc[abc_B])
        if i == 19:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] - [600, 0], vertices[j]])
        elif i == 21:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], [vertices[i] - [600, 0], vertices[j]])
        else:
            xc, yc = getCrossPoint([[x0, y0], [xn, yn]], vertices[[i, j]])
        equation, a, b, c = getLinearEquation([x0, y0], [xc, yc])
        print(equation)
        xCal = np.linspace(min(x0, xc), max(x0, xc), 50)
        yCal = getEquationResult(a, c, xCal)
        plt.plot(xCal, yCal, c='b')
        plt.plot(xCal + 600, yCal, c='b')
