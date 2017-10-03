# Preprocessing module


def map_col_to_ind(column):
    return column.astype('category').cat.codes.astype(int)


def map_str_to_ind(data, colname):
    return map_col_to_ind(data[colname])


def map_str_to_inds(data, colnames):
    for colname in colnames:
        data[colname] = map_str_to_ind(data, colname)
    return data
