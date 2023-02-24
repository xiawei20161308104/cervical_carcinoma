import os
root_dir = r'D:\code\Shanxi_data\data'
# 168个人
file_lists = os.listdir(root_dir)
i=0
for fi in file_lists:
    old=os.path.join(root_dir,str(fi))
    new=os.path.join(root_dir,str(i))
    os.rename(old,new)
    i+=1
file_lists = os.listdir(root_dir)

print(file_lists)