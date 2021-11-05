from img import Img
from img import LearningImg

# main ###
# init image ###
image = LearningImg(r"C:\Users\admin\PycharmProjects\courseWork\templates\road.jpg")
image1 = LearningImg(r"C:\Users\admin\PycharmProjects\courseWork\templates\forest.jpg")
image2 = LearningImg(r"C:\Users\admin\PycharmProjects\courseWork\templates\lake.jpg")

template = dict(
    road=image,
    forest=image1,
    lake=image2
)
# comparing of another classes pictures there
image.setForeign(image1)
image.setForeign(image2)

image1.setForeign(image)
image1.setForeign(image2)

image2.setForeign(image)
image2.setForeign(image1)


# learning stage
image.learning()
image1.learning()
image2.learning()

# class dependency charts
print(image.visualize(image1.name))
print(image.visualize(image2.name))
print(image1.visualize(image.name))
print(image1.visualize(image2.name))
print(image2.visualize(image.name))
print(image2.visualize(image1.name))

# recognition stage
testImg = Img(r"C:\Users\admin\PycharmProjects\courseWork\water.jpg")
testImg2 = Img(r"C:\Users\admin\PycharmProjects\courseWork\road.jpg")
testImg3 = Img(r"C:\Users\admin\PycharmProjects\courseWork\forest.jpg")
testImg.recognition(template)
testImg2.recognition(template)
testImg3.recognition(template)
