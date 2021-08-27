from vocab_processing import *

class BROWN_Processor:
    
    def __init__(self, brown_clusters_path, brown_size):

        self.brown_size = brown_size
        self.cluster_map = {}
        self.string_map = {}
        self.max_brown_cluster_index = 0
        self.cluster_n_map = [None]*(self.brown_size)
        self.unknown_map = {}

        index = 0
        string_map = {}
        
        for i in range(self.brown_size):
             self.cluster_n_map[i] = {}

        count = 1

        # Construct the clusters given an output file of words and their clusters.
        if brown_clusters_path != None:
            for line in open(brown_clusters_path):
                if len(line) > 0:
                    count += 1
                    line_data = line.split()
                    if len(line_data) > 1:
                        cluster = line_data[0]
                        word = line_data[1]
                        word = simplify_token(word)
                        cluster_num = index
                        
                        if cluster not in string_map:
                            self.cluster_map[word] = index
                            string_map[cluster] = index
                            index += 1
                        else:
                            cluster_num = string_map[cluster]
                            self.cluster_map[word] = cluster_num
                        for i in range(self.brown_size):
                            pref_id = index;
                            prefix = cluster[0: min(i+1, len(cluster))]
                            if prefix not in string_map:
                                string_map[prefix] = index
                                index += 1
                            else:
                                pref_id = string_map[prefix]
                            self.cluster_n_map[i][cluster_num] = pref_id
                if index > self.max_brown_cluster_index:
                    self.max_brown_cluster_index = index

        print("Brown Max: ", self.max_brown_cluster_index)

    # Retrieve the clusters of a given word.
    def get_brown_cluster(self, word):
        output = []
        if word == START_MARKER:
            return [-1]*(10*(self.brown_size+1))
        if word == END_MARKER:
            return [1]*(10*(self.brown_size+1))
        word = simplify_token(word)
        ids  = [None]*(self.brown_size+1)
        
        for i in range(0, self.brown_size+1):
            ids[i] = 0

        if word in self.cluster_map:
             ids[0] = self.cluster_map[word]
             
        if ids[0] > 0:
            for j in range(1, self.brown_size+1):
                ids[j] = self.cluster_n_map[j-1][ids[0]]

        for j in range(len(ids)):
            out1 = [int(i) for i in bin(ids[j])[2:]]
            out2 = [0]* (10 - len(out1)) + out1
            output.extend(out2)
        return output
