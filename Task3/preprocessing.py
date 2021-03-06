def map_col_to_ind(column):
    return column.astype('category').cat.codes.astype(int)


def map_to_ind(array):
    mapped = {}
    for idx, el in enumerate(array):
        mapped[el] = idx
    return mapped


def map_str_to_ind(data, colname):
    return map_col_to_ind(data[colname])


def map_str_to_inds(data, colnames):
    for colname in colnames:
        data[colname] = map_str_to_ind(data, colname) + 1
    return data


def get_tag_set(column, separator='|'):
    tag_set = set()
    for tags in column.astype(str):
        for tag in tags.split(separator):
            tag_set.add(tag)
    return tag_set


def get_tag_dict(column, separator='|'):
    tag_set = get_tag_set(column, separator)
    tag_dict = {}
    for idx, tag in enumerate(tag_set):
        tag_dict[tag] = idx
    return tag_dict


def normalize(column, min_max=(0.0, 1.0)):
    col_max = column.max()
    col_min = column.min()
    return column.apply(lambda x: ((x - col_min) / (col_max - col_min)) * (min_max[1] - min_max[0]) + min_max[0])
