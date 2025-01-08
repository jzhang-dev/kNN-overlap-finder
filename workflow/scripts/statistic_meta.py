#!python
import os  
  
# 输入文件夹路径  
folder_path = "./"  
  
# 输出文件名  
output_file = "summarize.csv"

for filename in os.listdir(folder_path):  
    if filename.endswith("stat.tsv"):  
        with open(os.path.join(folder_path, filename), 'r') as file:  
            for lines in file:
                line = lines.strip().split('\t')
                print(line[0])
                if line[0] == '0':
                    line = line[1:]
                    line.insert(0,filename)
                    new_line = '\t'.join(line) + '\n'
                    with open(output_file, 'a') as output:  
                        output.writelines(new_line)  