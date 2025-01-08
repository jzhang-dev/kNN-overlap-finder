import sys
sys.path.append("scripts")
    
regional_paf = open(snakemake.output["regional_paf"],'wt')
paf_file = snakemake.input["all_paf"]
chromosome, start, end = snakemake.params["region"]
chr_name = chromosome+'_MATERNAL'
for lines in open(paf_file,'rt'):
    line = lines.strip().split('\t')
    if len(line) < 6:
        print(line)
    else:
        if line[5] == chr_name and int(line[7]) > start and int(line[8]) < end:
            new = '\t'.join(line)
            regional_paf.write(new + '\n')  