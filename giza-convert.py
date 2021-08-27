import sys
import re

input_file = sys.argv[1]
reverse = sys.argv[2]

with open(input_file) as f:
    stats = ''
    for line in f:
        if line.startswith('# Sentence'):
            words = line.strip().split(' ')
            stats = words[6]+':'+words[9]+':'+words[13]
        if line.startswith('NULL'):
            alignments = []
            current = 0
            cols = re.compile('(\(\{|\}\))').split(line.strip())
            for i in range(6, len(cols), 4):
                indexes = cols[i].strip().split(' ')
                if len(indexes) > 0:
                    for index in indexes:
                        if len(index) > 0:
                            if reverse == 'N':
                                alignments.append(str(current)+'-'+str(int(index)-1))
                            elif reverse == 'Y':
                                alignments.append(str(int(index)-1)+'-'+str(current))
                current+=1   
            print(stats+' '+(' '.join(alignments)))

            
            
        
