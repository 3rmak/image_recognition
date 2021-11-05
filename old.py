from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

myu = 20


def center(nested_array_list):
    mean = np.mean(nested_array_list, axis=0)
    return mean


def to_binary(nested_array_list, center):
    copy_of_arr = nested_array_list.copy()
    for i in range(0, copy_of_arr.shape[2], 1):
        for j in range(0, copy_of_arr.shape[1], 1):
            for b in range(0, copy_of_arr.shape[0], 1):
                verh_dopusk = center[j][i] + myu
                niz_dopusk = center[j][i] - myu
                if (copy_of_arr[b][j][i] < verh_dopusk and copy_of_arr[b][j][i] > niz_dopusk):
                    copy_of_arr[b][j][i] = 1
                else:
                    copy_of_arr[b][j][i] = 0

    return copy_of_arr


def make_etalon(matrix):
    arr = np.sum(matrix, axis=0)
    # print(arr)
    for i in range(len(arr)):
        if(arr[i] > (matrix.shape[0] / 2)):
            arr[i] = 1
        else:
            arr[i] = 0
    return arr


# returns [difficalty estimate , D1 , beta, start of work zone , end of work zone, optimal radius]
def radius_selection(our, foreign):
    # our - from center to every in class realization array
    # foreign - from center to every in foreign class realization array
    estimate = []
    d1_arr = []
    beta_arr = []
    start_zone = 0
    end_zone = max(our) - 1
    radius = max(our)
    for r in range(radius):
        k1 = 0  # достоверность - к-во пойманных своих реализаций
        k2 = 0  # ошибка - к-во пойманых "соседних" реализаций
        for i in range(len(our)):
            if(our[i] <= r):
                k1 += 1
            if(foreign[i] <= r):
                k2 += 1
        D1 = k1/len(our)
        beta = k2/len(foreign)
        if (D1 < 0.5):
            start_zone = r + 1
        if(beta > 0.5 and end_zone == max(our) - 1):
            end_zone = r
        diff = D1 - beta
        kriterij_kulbaka = diff * math.log((1 + diff + 0.1) / (1 - diff + 0.1)) / math.log(2)
        estimate.append(kriterij_kulbaka)
        d1_arr.append(D1)
        beta_arr.append(beta)

    return [estimate, d1_arr, beta_arr, start_zone, end_zone, estimate.index(max(estimate))]


# show chart
def visualize(arr, str):

    plt.xlabel('Radius')
    plt.ylabel('Value')
    plt.title(str)

    plt.plot(arr[0], label='criteria', linewidth=3)
    plt.plot(arr[1], label='D1 - own')
    plt.plot(arr[2], label='beta - neighs')
    plt.plot(100, 5)
    plt.axvspan(arr[3], arr[4], alpha=0.3, color='red', label='working zone')
    plt.hlines(0.5, 0, 100, color='red', linestyles='dashed', label='border estimate for D1')
    plt.legend(loc='upper right')
    plt.show()


def recognition(vector):
    # is hamming distance between center of new class
    # and known class less than known radius
    if np.count_nonzero(vector != etalonV) <= optimal_radius_road:
        return "img is ROAD"
    elif np.count_nonzero(vector != etalonV2) <= optimal_radius_forest:
        return "img is FOREST"
    elif np.count_nonzero(vector != etalonV2) <= optimal_radius_water:
        return "img is WATER"
    else:
        return "img is UNKNOWN"


##################################################################################
# learning #######################################################################
##################################################################################
# load the image
image = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\road.jpg")
image2 = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\forest.jpg")
image3 = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\water.jpg")


# convert image to numpy array
data = np.asarray(image)
data2 = np.asarray(image2)
data3 = np.asarray(image3)

# center vector
centerVector = center(data)
centerVector2 = center(data2)
centerVector3 = center(data3)

# binary Matrix
binaryMatrix = to_binary(data, centerVector)
binaryMatrix2 = to_binary(data2, centerVector2)
binaryMatrix3 = to_binary(data3, centerVector3)

# new mapping . (3, 9) instead (3, 3, 3)
binaryMatrix = binaryMatrix.reshape((data.shape[0], data.shape[1] * data.shape[2]))
binaryMatrix2 = binaryMatrix2.reshape((data2.shape[0], data2.shape[1] * data2.shape[2]))
binaryMatrix3 = binaryMatrix3.reshape((data3.shape[0], data3.shape[1] * data3.shape[2]))

# etalon vector
etalonV = make_etalon(binaryMatrix)
etalonV2 = make_etalon(binaryMatrix2)
etalonV3 = make_etalon(binaryMatrix3)

# distances beetwen
# class center and every own class realization
arrEtalonV_realiz = []
for i in range(binaryMatrix.shape[0]):
    arrEtalonV_realiz.append(np.count_nonzero(binaryMatrix[i] != etalonV))

arrEtalonV_realiz2 = []
for i in range(binaryMatrix2.shape[0]):
    arrEtalonV_realiz2.append(np.count_nonzero(binaryMatrix2[i] != etalonV2))

arrEtalonV_realiz3 = []
for i in range(binaryMatrix3.shape[0]):
    arrEtalonV_realiz3.append(np.count_nonzero(binaryMatrix3[i] != etalonV3))

##################################################################################
# array of radius
# FROM etalon vector
# TO every foreign class realization

# center - road , realiz - forest
arrForeignEtalonV_realiz1 = []
for i in range(binaryMatrix.shape[0]):
    arrForeignEtalonV_realiz1.append(np.count_nonzero(binaryMatrix[i] != etalonV2))

# center - road , realiz - water
arrForeignEtalonV_realiz11 = []
for i in range(binaryMatrix.shape[0]):
    arrForeignEtalonV_realiz11.append(np.count_nonzero(binaryMatrix[i] != etalonV3))

# center - forest , realiz - road
arrForeignEtalonV_realiz2 = []
for i in range(binaryMatrix2.shape[0]):
    arrForeignEtalonV_realiz2.append(np.count_nonzero(binaryMatrix2[i] != etalonV))

# center - forest , realiz - water
arrForeignEtalonV_realiz22 = []
for i in range(binaryMatrix2.shape[0]):
    arrForeignEtalonV_realiz22.append(np.count_nonzero(binaryMatrix2[i] != etalonV3))

# center - water , realiz - road
arrForeignEtalonV_realiz3 = []
for i in range(binaryMatrix3.shape[0]):
    arrForeignEtalonV_realiz3.append(np.count_nonzero(binaryMatrix3[i] != etalonV))

# center - water , realiz - forest
arrForeignEtalonV_realiz33 = []
for i in range(binaryMatrix3.shape[0]):
    arrForeignEtalonV_realiz33.append(np.count_nonzero(binaryMatrix3[i] != etalonV2))


radius_road = radius_selection(arrEtalonV_realiz, arrForeignEtalonV_realiz1)
optimal_radius_road = radius_road[5]
radius_forest = radius_selection(arrEtalonV_realiz2, arrForeignEtalonV_realiz2)
optimal_radius_forest = radius_forest[5]
radius_water = radius_selection(arrEtalonV_realiz3, arrForeignEtalonV_realiz3)
optimal_radius_water = radius_water[5]

# charts #########################################################################
visualize(radius_selection(arrEtalonV_realiz, arrForeignEtalonV_realiz1)
          , 'center - road , realiz - forest')

# visualize(radius_selection(arrEtalonV_realiz, arrForeignEtalonV_realiz11)
#           , 'center - road , realiz - water')
#
# visualize(radius_selection(arrEtalonV_realiz2, arrForeignEtalonV_realiz2)
#           , 'center - forest , realiz - road')
#
# visualize(radius_selection(arrEtalonV_realiz2, arrForeignEtalonV_realiz22)
#           , 'center - forest , realiz - water')
#
# visualize(radius_selection(arrEtalonV_realiz3, arrForeignEtalonV_realiz3)
#           , 'center - water , realiz - road')
#
# visualize(radius_selection(arrEtalonV_realiz3, arrForeignEtalonV_realiz33)
#           , 'center - water , realiz - forest')

##################################################################################
# new images recognition #########################################################
##################################################################################

test_img = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\test\test.jpg")
test_img2 = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\test\test2.jpg")
test_img3 = Image.open(r"C:\Users\admin\PycharmProjects\courseWork\test\test3.jpg")

test_data = np.asarray(test_img)
test_data2 = np.asarray(test_img2)
test_data3 = np.asarray(test_img3)

# center vector
testCenterVector = center(test_data)
testCenterVector2 = center(test_data2)
testCenterVector3 = center(test_data3)

# binary Matrix
testBinaryMatrix = to_binary(test_data, testCenterVector)
testBinaryMatrix2 = to_binary(test_data2, testCenterVector2)
testBinaryMatrix3 = to_binary(test_data3, testCenterVector3)

# new mapping . (3, 9) instead (3, 3, 3)
testBinaryMatrix = testBinaryMatrix.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
testBinaryMatrix2 = testBinaryMatrix2.reshape((test_data2.shape[0], test_data2.shape[1] * test_data2.shape[2]))
testBinaryMatrix3 = testBinaryMatrix3.reshape((test_data3.shape[0], test_data3.shape[1] * test_data3.shape[2]))

# etalonV - road
# etalonV2 - forest
# etalonV3 - water

# etalon vector
testEtalonV = make_etalon(testBinaryMatrix)
testEtalonV2 = make_etalon(testBinaryMatrix2)
testEtalonV3 = make_etalon(testBinaryMatrix3)

print(recognition(testEtalonV))
print(recognition(testEtalonV2))
print(recognition(testEtalonV3))
