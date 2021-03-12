from pprint import pprint

def calc_pog(master, before_info, after_info):
    f0 = before_info['section_1']['front_label']
    f1 = after_info['section_1']['front_label']
    
    b0 = before_info['section_1']['total_count'] - before_info['section_1']['front_count']
    b1 = after_info['section_1']['total_count'] - after_info['section_1']['front_count']
    
    D = {}
    for i in f0:
        D[i] = D.get(i, 0) + 1
    for i in f1:
        D[i] = D.get(i, 0) - 1
    print(D)
    D[master] = D.get(master, 0) + b0 - b1
    print(D)
    return D

master = 'blue'
before = {'direction': 'right',
        'floor': '1',
        'image_name': './test_images_ht/ht_1_right.jpg',
        'section_1': {'front_corr': [(148, 862, 854, 947),
                                    (149, 763, 858, 848),
                                    (176, 679, 832, 754)],
                    'front_count': 3,
                    'front_label': ['blue', 'blue', 'amber'],
                    'total_corr': [(148, 862, 854, 947),
                                    (149, 763, 858, 848),
                                    (257, 376, 743, 419),
                                    (214, 536, 793, 597),
                                    (281, 291, 715, 324),
                                    (289, 253, 703, 282),
                                    (273, 331, 725, 369),
                                    (236, 423, 766, 469),
                                    (176, 679, 832, 754),
                                    (222, 477, 779, 529),
                                    (194, 603, 815, 670)],
                    'total_count': 11},
        'total_count': 13}

after = {'direction': 'right',
 'floor': '1',
 'image_name': './test_images_ht/ht_1_right.jpg',
 'section_1': {'front_corr': [(148, 862, 854, 947),
                              (149, 763, 858, 848),
                              (176, 679, 832, 754)],
               'front_count': 3,
               'front_label': ['red', 'red', 'red'],
               'total_corr': [(148, 862, 854, 947),
                              (149, 763, 858, 848),
                              (257, 376, 743, 419),
                              (214, 536, 793, 597),
                              (281, 291, 715, 324),
                              (289, 253, 703, 282),
                              (273, 331, 725, 369),
                              (236, 423, 766, 469),
                              (176, 679, 832, 754),
                              (222, 477, 779, 529),
                              (194, 603, 815, 670)],
               'total_count': 11},
 'total_count': 13}

calc_pog(master = master, before_info = before, after_info = after)