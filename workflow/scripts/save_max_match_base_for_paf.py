
#!/usr/bin/env python3
import sys
from collections import defaultdict

def filter_best_match(input_paf, output_paf):
    # 存储每个read的最佳比对记录
    best_matches = defaultdict(dict)
    
    with open(input_paf, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 12:  # 确保是有效的PAF行
                continue
                
            qname = fields[0]      # 查询read名称
            nmatch = int(fields[9]) # 匹配碱基数(第10列)
            alen = int(fields[10])  # 比对长度(第11列)
            mapq = int(fields[11]) # 比对质量(第12列)
            
            # 只保留每个read的nmatch最大的记录
            if qname not in best_matches or nmatch > best_matches[qname]['nmatch']:
                best_matches[qname] = {
                    'line': line,
                    'nmatch': nmatch,
                    'alen': alen,
                    'mapq': mapq
                }
    
    # 写入最佳比对结果
    with open(output_paf, 'w') as out:
        for record in best_matches.values():
            out.write(record['line'])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.paf> <output.paf>")
        sys.exit(1)
        
    input_paf = sys.argv[1]
    output_paf = sys.argv[2]
    filter_best_match(input_paf, output_paf)