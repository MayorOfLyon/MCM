import os
import shutil

def duplicate_images(input_folder, output_folder, n):
    # 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的每个文件
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # 检查文件是否是一个普通文件
        if os.path.isfile(filepath):
            # 复制文件 n 次到目标文件夹
            for i in range(n):
                # 构造新的文件名，例如：original_filename_1.jpg
                new_filename = f"{os.path.splitext(filename)[0]}_{i+1}{os.path.splitext(filename)[1]}"
                new_filepath = os.path.join(output_folder, new_filename)
                
                # 复制文件
                shutil.copyfile(filepath, new_filepath)

# 例子：将原始文件夹中的图片复制3次到目标文件夹
input_folder = './excel_data/positive'
output_folder = './output_folder/train/positive'
n = 20
duplicate_images(input_folder, output_folder, n)