#!python
import os  
import sys
# 输入文件夹路径  
folder_path = ['./CHM13/HLA','./CHM13/IGK','./yeast/chr10','./TAIR/chr3','./human/HLA','./human/IGK','./human/chr22','./human/chr1']
# 输出文件名  
output_file = "summarize_stat_n6.csv"
output_file4 = "summarize_stat_all_n.csv"
output_file3 = "summarize_stat_n12.csv"
output_file2 = "summarize_benchmark.csv"

# 遍历文件夹  
for folder in folder_path:
    for dirname1 in os.listdir(folder):
        dir_path = os.path.join(folder, dirname1,'kmer_k16')
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith("overlap_stat.tsv"): 
                    with open(os.path.join(dir_path, filename), 'r') as file:  
                        for lines in file:
                            line = lines.strip().split('\t')
                            if len(line) > 2 :
                                if 'precision' not in lines:
                                    with open(output_file4, 'a') as output: 
                                        output.write(folder + "\t" + dirname1 + "\t") 
                                        output.writelines(lines)  
                                if line[2] == str(6):
                                    with open(output_file, 'a') as output: 
                                        output.write(folder + "\t" + dirname1 + "\t")  # 写入文件名  
                                        output.writelines(lines)  
                                elif line[2] == str(12):
                                    with open(output_file3, 'a') as output:
                                        output.write(folder + "\t" + dirname1 + "\t") 
                                        output.writelines(lines) 
                if filename.endswith("benchmark.csv"):  
                    with open(os.path.join(dir_path, filename), 'r') as file:  
                        lines = file.readlines()  
                        extracted_lines = lines[1]
                        # 写入新文件  
                        with open(output_file2, 'a') as output:  
                            output.write(folder + "\t" + dirname1 + "\t" + filename + "\t")  # 写入文件名  
                            output.writelines(extracted_lines)  
