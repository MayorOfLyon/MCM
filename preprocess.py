import pandas as pd
import warnings
import os
import shutil
warnings.filterwarnings('ignore')

# read data
image_table = pd.read_excel('./excel_data/image_data.xlsx')
information_table = pd.read_excel('./excel_data/information.xlsx')

merged_df = pd.merge(image_table, information_table, on='GlobalID', how='inner')
new_data = merged_df[['Lab Status', 'FileName']]

grouped_data = new_data.groupby('Lab Status')

separated_dataframes = []
for status, group in grouped_data:
    print(status)
    separated_dataframes.append(group[['FileName']].copy())
    
negative_df = separated_dataframes[0]
positive_df = separated_dataframes[1]
unprocessed_df = separated_dataframes[2]
unverified_df = separated_dataframes[3]

# remove images
output_root = './output_folder/'
os.makedirs(output_root, exist_ok=True)

for df, folder_name in zip([negative_df, positive_df, unprocessed_df, unverified_df], ['negative', 'positive', 'unprocessed', 'unverified']):
    # 创建子文件夹
    output_folder = os.path.join(output_root, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        # 获取文件名
        filename = row['FileName']

        # 构造源文件路径和目标文件路径
        source_path = os.path.join('./dat/data', filename)  # 替换为你的图片文件夹路径
        target_path = os.path.join(output_folder, filename)

        # 使用 shutil.move 将文件移动到目标文件夹
        shutil.move(source_path, target_path)

print("Files moved successfully.")