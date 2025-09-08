import pandas as pd
import sys

tsv_file = sys.argv[1]
out_filter_tsv = sys.argv[2]
read_name_file = sys.argv[3]

# 读取TSV文件
df = pd.read_table(tsv_file, sep='\t')

# 过滤read_length大于5000的行
df_filter = df[df['read_length'] > 5000]

# 保存过滤后的数据到TSV
df_filter.to_csv(out_filter_tsv, sep='\t', index=False)

# 获取唯一的read_name并转换为字符串
read_name_list = set(df_filter['read_name'].astype(str))
sort_read_name_list = sorted(read_name_list)
# 写入read_name到文件
with open(read_name_file, 'w') as f:
    for name in sort_read_name_list:
        f.write(f"{name}\n")  # 使用f-string确保换行符正确