import os
dataset_guipang= '/repository/gong/qiafan/guipangdata/'

# print(os.path.join(dataset_guipang,'train')
# for filename in os.listdir(os.path.join(dataset_guipang,'train')):
#     if os.path.splitext(filename)[1]=='.jpg':
#         print(filename)
images=[]
annotations=[]

for filename in os.listdir(os.path.join(dataset_guipang,'train')):
    if os.path.splitext(filename)[1]=='.jpg':
        images.append(filename)
        annotations.append(os.path.splitext(filename)[0]+'.xml')

    # elif os.path.splitext(filename)[1]=='.xml':
    #     annotations.append(filename)
print(images)
print(annotations)