import os

image_files = []
os.chdir(os.path.join("data", "custom", "images"))
#os.chdir(os.path.join("data", "train_3D"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append("data/custom/images/" + filename)
#        image_files.append("data/train_3D/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..") 
