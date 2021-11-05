from PIL import Image
import matplotlib.pyplot as plt
from radius import Radius
import os.path
import numpy as np
import math

myu = 15


class Img:
    def __init__(self, img_path):
        if(os.path.exists(img_path)):
            self.img = Image.open(img_path)
            self.img = self.img.resize((50, 50))
            self.name = os.path.basename(img_path)
            self.imgToData = np.asarray(self.img)
            self.centerVector = self.center()
            self.binaryMatrix = self.binarization()
            self.etalonVector = self.calculateEtalon()
            self.own_realization = self.getRadiuses(self.etalonVector)

    imgToData = []
    centerVector = []
    binaryMatrix = []
    etalonVector = []

    def center(self):
      return np.mean(self.imgToData, axis=0)

    def binarization(self):
        copy_of_arr = self.imgToData.copy()
        for i in range(0, copy_of_arr.shape[2], 1):
            for j in range(0, copy_of_arr.shape[1], 1):
                for b in range(0, copy_of_arr.shape[0], 1):
                    verh_dopusk = self.centerVector[j][i] + myu
                    niz_dopusk = self.centerVector[j][i] - myu
                    if (copy_of_arr[b][j][i] < verh_dopusk and
                            copy_of_arr[b][j][i] > niz_dopusk):
                        copy_of_arr[b][j][i] = 1
                    else:
                        copy_of_arr[b][j][i] = 0
        # return new size matrix
        return copy_of_arr.reshape((self.imgToData.shape[0],
                                    self.imgToData.shape[1] * self.imgToData.shape[2]))

    def calculateEtalon(self):
        arr = np.sum(self.binaryMatrix, axis=0)
        # print(arr)
        for i in range(len(arr)):
            if (arr[i] > (self.binaryMatrix.shape[0] / 2)):
                arr[i] = 1
            else:
                arr[i] = 0
        return arr

    def getRadiuses(self, vector):
        radiuses = []
        for i in range(self.binaryMatrix.shape[0]):
            radiuses.append(np.count_nonzero(self.binaryMatrix[i] != vector))
        return radiuses

    def recognition(self, template):
        # hamming distance between center of new class
        # and known class less than known radius
        # and smaller than other classes radius
        class_name = 'UNKNOWN'
        smallest_distance = 1000
        for item in template.values():
            radius = np.count_nonzero(self.etalonVector != item.etalonVector)
            if (radius <= item.class_radius.radius) and (radius < smallest_distance):
                class_name = item.name
                smallest_distance = radius
        print(f"{self.name} is {class_name}")


class LearningImg(Img):

    class_radius = Radius(0, 0)
    own_realization = []
    foreign_realizations = dict()
    graphic_params = dict()    # dictionary with chart parameters
                              # 0 - kulback criteria
                              # 1 - D1 criteria
                              # 2 - beta criteria
                              # 3 - start working zone
                              # 4 - end working zone

    def setForeign(self, another):
        self.foreign_realizations["%s" % another.name] = self.getRadiuses(another.etalonVector)

    def learning(self):
        # our - from center to every in class realization array
        # foreign - from center to every in foreign class realization array
        for iterator in (self.foreign_realizations.keys()):
            estimate = []
            d1_arr = []
            beta_arr = []
            start_zone = 0
            end_zone = max(self.own_realization) - 1
            radius = max(self.own_realization)
            for r in range(radius):
                k1 = 0  # достоверность - к-во пойманных своих реализаций
                k2 = 0  # ошибка - к-во пойманых "соседних" реализаций
                for i in range(len(self.own_realization)):
                    if (self.own_realization[i] <= r):
                        k1 += 1
                    if (self.foreign_realizations[iterator][i] <= r):
                        k2 += 1
                D1 = k1 / len(self.own_realization)
                beta = k2 / len(self.foreign_realizations[iterator])
                if (D1 < 0.5):
                    start_zone = r + 1
                if (beta > 0.5 and end_zone == max(self.own_realization) - 1):
                    end_zone = r
                diff = D1 - beta
                kriterij_kulbaka = diff * math.log((1 + diff + 0.1) / (1 - diff + 0.1)) / math.log(2)
                estimate.append(kriterij_kulbaka)
                d1_arr.append(D1)
                beta_arr.append(beta)

            radius = Radius(max(estimate), estimate.index(max(estimate)))
            self.class_radius.set_radius(radius)
            self.graphic_params["%s" % iterator] = dict(
                kulback=estimate,
                D1Arr=d1_arr,
                betaArr=beta_arr,
                startZone=start_zone,
                endZone=end_zone
            )
        print("Learning of " + self.name + " ends successful")

    def visualize(self, item):
        plt.xlabel('Radius')
        plt.ylabel('Value')
        plt.title(f"Центр - {self.name}\nРеализации - {item}")

        plt.plot(self.graphic_params[item]['kulback'], label='criteria', linewidth=3)
        plt.plot(self.graphic_params[item]['D1Arr'], label='D1 - own')
        plt.plot(self.graphic_params[item]['betaArr'], label='beta - neighs')
        plt.plot(100, 5)
        plt.axvspan(self.graphic_params[item]['startZone'], self.graphic_params[item]['endZone'], alpha=0.3, color='red', label='working zone')
        plt.hlines(0.5, 0, 100, color='red', linestyles='dashed', label='border estimate for D1')
        plt.legend(loc='upper right')
        plt.show()
