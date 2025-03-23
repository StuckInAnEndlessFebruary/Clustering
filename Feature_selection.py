import numpy as np
import csv
FEATURE_PATH = "features.csv"

data_array = []
mean_feature = []
def find_correlation():


    # Read the CSV file
    with open(FEATURE_PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            data_array.append(row)
    
    #test
    # print (len(data_array))
    # print(len(data_array[0]))

    #save mean of each feature in mean_feature
    for i in range(len(data_array[0])-1):
        mean = 0
        for j in range(1 , len(data_array)):
            mean += float(data_array[j][i])
            #test
            # print(mean)
        
        mean_feature.append(mean / len(data_array)-1)

    
    correlation_matrix = np.zeros((len(data_array[0])-1,len(data_array[0])-1))
    for i in range(len(data_array[0])-1):
        for j in range(i+1 , len(data_array[0])-1):
            sorat = 0
            x_pow2 =0
            y_pow2 = 0
            for k in range(1,len(data_array)):
                sorat += (float(data_array[k][i]) - float(mean_feature[i]))*(float(data_array[k][j]) - float(mean_feature[j]))
                x_pow2 += pow((float(data_array[k][i]) - float(mean_feature[i])),2)
                y_pow2 += pow((float(data_array[k][j]) - float(mean_feature[j])),2)

            makhraj = (pow((x_pow2 * y_pow2),1/2))
            if makhraj != 0:
                # correlation_matrix[i][j] = sorat /makhraj
                
                correlation_matrix[i][j] = round(sorat /makhraj,2)
            else:
                correlation_matrix[i][j] = 0
            correlation_matrix[j][i] = correlation_matrix[i][j]


    #     # Save the correlation matrix to a txt file
    # with open("correlation_matrix.txt", "w") as file:
    #     for row in correlation_matrix:
    #         row_str = "\t".join(map(str, row)) 
    #         file.write(row_str + "\n") 

    print (f"*****mean_feature*****\n {mean_feature}")
    print(f"\n\n********** correlation_matrix***********\n {correlation_matrix}")
    return correlation_matrix

# def select_least_k_features(correlation_matrix, k):
#     features = np.zeros((k, 3)) 

#     #all diffrent corrolation
#     all_correlatoin = []
#     num_features = len(correlation_matrix)
#     for i in range(num_features):
#         for j in range(i + 1, num_features):
#             all_correlatoin.append((abs(correlation_matrix[i][j]), i, j))

#     #test 
#     # print(all_correlatoin)

#     # Sort asc
#     all_correlatoin.sort(key=lambda x: x[0])

#     #test
#     print(all_correlatoin)

#     # Select the top k 
#     for idx in range(k):
#         features[idx, 0] = all_correlatoin[idx][0]
#         features[idx, 1] = all_correlatoin[idx][1]
#         features[idx, 2] = all_correlatoin[idx][2]
#     print (features)
#     return features
import numpy as np

def select_least_k_features(correlation_matrix, k):
    num_features = len(correlation_matrix)
    selected_features = [] 

    # Sum of absolute correlations for each feature
    scores = np.sum(np.abs(correlation_matrix), axis=1)  

    for _ in range(k):
        best_feature = None
        best_score = float('inf')

        for feature in range(num_features):
            if feature not in selected_features and scores[feature] < best_score:
                best_feature = feature
                best_score = scores[feature]

        selected_features.append(best_feature)

        for feature in range(num_features):
            if feature not in selected_features: 
                total_correlation = 0 
                for selected in selected_features:
                    total_correlation += abs(correlation_matrix[feature][selected])
                scores[feature] = total_correlation

    print (selected_features)
    return selected_features

select_least_k_features(find_correlation() , 3)