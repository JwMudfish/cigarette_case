from pprint import pprint

def calc_pog(master, before_info, after_info):
    f0 = before_info['section_1']['front_result']
    f1 = after_info['section_1']['front_result']
    
    b0 = before_info['section_1']['total_count'] - before_info['section_1']['front_count']
    b1 = after_info['section_1']['total_count'] - after_info['section_1']['front_count']
    
    D = {}
    for i in f0:
        D[i] = D.get(i, 0) + 1
    for i in f1:
        D[i] = D.get(i, 0) - 1
    #print(D)
    D[master] = D.get(master, 0) + b0 - b1
    print(D)
    return D

master = 'green'
before = {'direction': 'right',
 'floor': '1',
 'image_name': './test_images_cnt/before_1_right.jpg',
 'section_1': {'front_corr': [(172, 851, 871, 945),
                              (143, 756, 896, 837),
                              (166, 671, 870, 745)],
               'front_count': 3,
               'front_result': ['green', 'green', 'green'],
               'total_corr': [(172, 851, 871, 945),
                              (143, 756, 896, 837),
                              (166, 671, 870, 745),
                              (243, 468, 789, 518),
                              (204, 594, 834, 660),
                              (225, 527, 811, 586),
                              (264, 413, 767, 458)],
               'total_count': 7},
 'total_count': 9}

after = {'direction': 'right',
 'floor': '1',
 'image_name': './test_images_cnt/after_1_right.jpg',
 'section_1': {'front_corr': [(129, 847, 913, 938),
                              (148, 754, 893, 833),
                              (185, 668, 852, 742)],
               'front_count': 3,
               'front_result': ['green', 'green', 'green'],
               'total_corr': [(129, 847, 913, 938),
                              (185, 668, 852, 742),
                              (148, 754, 893, 833),
                              (231, 523, 802, 579),
                              (216, 590, 820, 656)],
               'total_count': 5},
 'total_count': 5}

calc_pog(master = master, before_info = before, after_info = after)