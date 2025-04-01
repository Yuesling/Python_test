import csv

with open('score.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for data_list in reader:

        print(reader.line_num, end='\t')
        for ele in data_list:
            print(ele, end='\t')
        print()
