import numpy as np


def creat_coarse_data(mg_dict, dataset_train, dataset_test):

    train_coarse_array_multi = []

    for gran in mg_dict:
        dataset_train_coarse = []
        dataDict_train_coarse = {}

        for dataDict in dataset_train:
            # reshape the array to have n columns
            coarse_array = []
            for item in dataDict['target']:
                max_gran = max([int(gran_cur) for gran_cur in mg_dict])
                trancated_length_train = len(item) - len(item) % max_gran
                arr = item[0:trancated_length_train].reshape(-1, int(gran))
                # sum every n elements of the array
                sum_arr = np.mean(arr, axis=1)
                sum_arr_align = np.repeat(sum_arr, int(gran))
                coarse_array.append(sum_arr_align)
                # print(sum_arr_align)
                train_coarse_array_multi.append(sum_arr_align)

    dataDict_train_coarse['target'] = np.array(train_coarse_array_multi)
    dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
    dataDict_train_coarse['start'] = dataDict['start']
    dataset_train_coarse.append(dataDict_train_coarse)

    # dataDict_test_coarse = {}
    # create a new dictionary for the coarse-grained dataset, should be put under the for loop
    dataset_test_coarse = []

    for dataDict in dataset_test:
        dataDict_test_coarse = {}  # create a new dictionar
        test_coarse_array_multi = []
        for gran in mg_dict:
            for item in dataDict['target']:
                # print(len(item))
                # calculate the index can be divided by gran
                gran_num = int(gran)
                cut_index = len(item) - len(item) % gran_num
                arr_before = item[:cut_index].reshape(-1, gran_num)
                arr_before_mean = np.mean(arr_before, axis=1)
                mean_arr_align = np.repeat(arr_before_mean, int(gran))
                if len(item) % gran_num != 0:
                    arr_after = item[cut_index:].reshape(
                        -1, len(item) % gran_num)
                    arr_after_mean = np.mean(arr_after, axis=1)
                    # sum every 4 elements of the array
                    mean_arr_align_after = np.repeat(
                        arr_after_mean, len(item) % gran_num)
                    mean_arr_align = np.concatenate(
                        (mean_arr_align, mean_arr_align_after), axis=0)
                # print(len(sum_arr_align))
                test_coarse_array_multi.append(mean_arr_align)
                # print(len(coarse_array))
            dataDict_test_coarse['target'] = np.array(test_coarse_array_multi)
            dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
            dataDict_test_coarse['start'] = dataDict['start']
        dataset_test_coarse.append(dataDict_test_coarse)
    # print(dataset.metadata.prediction_length)
    data_train = dataset_train_coarse
    data_test = dataset_test_coarse

    return data_train, data_test


def creat_coarse_data_elec(mg_dict, dataset_train, dataset_test):

    train_coarse_array_multi = []
    max_gran = max([int(cur_gran) for cur_gran in mg_dict])
    index_trun_start = 1 if max_gran != 48 else 25
    for gran in mg_dict:
        dataset_train_coarse = []
        dataset_test_coarse = []

        dataDict_train_coarse = {}
        for dataDict in dataset_train:
            # reshape the array to have n columns
            coarse_array = []
            for item in dataDict['target']:

                arr = item[index_trun_start:5833].reshape(-1, int(gran))
                # sum every n elements of the array
                sum_arr = np.mean(arr, axis=1)
                sum_arr_align = np.repeat(sum_arr, int(gran))
                coarse_array.append(sum_arr_align)
                # print(sum_arr_align)
                train_coarse_array_multi.append(sum_arr_align)

            dataDict_train_coarse['target'] = np.array(
                train_coarse_array_multi)
            dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
            dataDict_train_coarse['start'] = dataDict['start']
            dataset_train_coarse.append(dataDict_train_coarse)

    dataDict_test_coarse = {}
    dataset_test_coarse = []
    for dataDict in dataset_test:
        # reshape the array to have 4 columns
        coarse_array = []
        for gran in mg_dict:
            dataDict_train_coarse = {}
            for item in dataDict['target']:
                # calculate the index can be divided by gran
                gran_num = int(gran)
                cut_index = len(item) - len(item) % gran_num
                arr_before = item[:cut_index].reshape(-1, gran_num)
                arr_before_mean = np.mean(arr_before, axis=1)
                mean_arr_align = np.repeat(arr_before_mean, int(gran))
                if len(item) % gran_num != 0:
                    arr_after = item[cut_index:].reshape(
                        -1, len(item) % gran_num)
                    arr_after_mean = np.mean(arr_after, axis=1)
                    # sum every 4 elements of the array
                    mean_arr_align_after = np.repeat(
                        arr_after_mean, len(item) % gran_num)
                    mean_arr_align = np.concatenate(
                        (mean_arr_align, mean_arr_align_after), axis=0)
                coarse_array.append(mean_arr_align)
        dataDict_test_coarse['target'] = np.array(coarse_array)
        dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_test_coarse['start'] = dataDict['start']
        dataset_test_coarse.append(dataDict_test_coarse)
    data_train = dataset_train_coarse
    data_test = dataset_test_coarse

    return data_train, data_test
