#!python
import os  
  
# 输入文件夹路径  
folder_path = "./"  
  
# 输出文件名  
output_file = "summarize_stat_n6.csv"
  
# 遍历文件夹  
for filename in os.listdir(folder_path):  
    if filename.endswith("overlap_stat.tsv"):  
        with open(os.path.join(folder_path, filename), 'r') as file:  
            for lines in file:
                line = lines.strip().split('\t')
                if len(line) > 2 and line[2] == str(6):
                    with open(output_file, 'a') as output:  
                        output.writelines(lines)  

output_file3 = "summarize_stat_n12.csv"
  
# 遍历文件夹  
for filename in os.listdir(folder_path):  
    if filename.endswith("overlap_stat.tsv"):  
        with open(os.path.join(folder_path, filename), 'r') as file:  
            for lines in file:
                line = lines.strip().split('\t')
                if len(line) > 2 and line[2] == str(12):
                    with open(output_file3, 'a') as output:  
                        output.writelines(lines) 

output_file2 = "summarize_benchmark.csv"
  
# 遍历文件夹  
for filename in os.listdir(folder_path):  
    if filename.endswith("benchmark.csv"):  
        with open(os.path.join(folder_path, filename), 'r') as file:  
            lines = file.readlines()  
            # 提取第6和12行  
            extracted_lines = lines[1]
            # 写入新文件  
            with open(output_file2, 'a') as output:  
                output.write(filename + "\t")  # 写入文件名  
                output.writelines(extracted_lines)  
