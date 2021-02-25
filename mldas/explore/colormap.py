__copyright__ = """
Machine Learning for Distributed Acoustic Sensing data (MLDAS)
Copyright (c) 2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
"""
__license__ = "Modified BSD license (see LICENSE.txt)"
__maintainer__ = "Vincent Dumont"
__email__ = "vincentdumont11@gmail.com"

import numpy

def custom_cm():
    """
    Custom colormap for parallel coordinate plot. The colormap was
    created using `this website<https://jdherman.github.io/colormap/>`_.
    """
    cm = numpy.array([
         [0,0,0],
         [2,0,0],
         [4,0,0],
         [6,0,0],
         [8,0,0],
         [10,0,0],
         [13,0,0],
         [15,0,0],
         [17,0,0],
         [19,0,0],
         [21,0,0],
         [23,0,0],
         [25,0,0],
         [27,0,0],
         [29,0,0],
         [31,0,0],
         [33,0,0],
         [35,0,0],
         [38,0,0],
         [40,0,0],
         [42,0,0],
         [44,0,0],
         [46,0,0],
         [48,0,0],
         [50,0,0],
         [52,0,0],
         [54,0,0],
         [56,0,0],
         [58,0,0],
         [60,0,0],
         [63,0,0],
         [65,0,0],
         [67,0,0],
         [69,0,0],
         [71,0,0],
         [73,0,0],
         [75,0,0],
         [77,0,0],
         [79,0,0],
         [81,0,0],
         [83,0,0],
         [85,0,0],
         [88,0,0],
         [90,0,0],
         [92,0,0],
         [94,0,0],
         [96,0,0],
         [98,0,0],
         [100,0,0],
         [102,0,0],
         [104,0,0],
         [106,0,0],
         [108,0,0],
         [110,0,0],
         [113,0,0],
         [115,0,0],
         [117,0,0],
         [119,0,0],
         [121,0,0],
         [123,0,0],
         [125,0,0],
         [127,0,0],
         [129,0,0],
         [131,0,0],
         [133,0,0],
         [135,1,0],
         [136,2,0],
         [138,3,0],
         [140,4,0],
         [141,5,0],
         [143,6,0],
         [144,7,0],
         [146,8,0],
         [148,9,0],
         [149,10,0],
         [151,11,0],
         [152,12,0],
         [154,13,0],
         [156,14,0],
         [157,15,0],
         [159,16,0],
         [160,17,0],
         [162,18,0],
         [163,19,0],
         [165,20,0],
         [167,21,0],
         [168,22,0],
         [170,23,0],
         [171,24,0],
         [173,25,0],
         [175,26,0],
         [176,27,0],
         [178,28,0],
         [179,29,0],
         [181,30,0],
         [183,31,0],
         [184,32,0],
         [186,33,0],
         [187,34,0],
         [189,35,0],
         [190,36,0],
         [192,37,0],
         [194,38,0],
         [195,39,0],
         [197,40,0],
         [198,41,0],
         [200,42,0],
         [202,43,0],
         [203,44,0],
         [205,45,0],
         [206,46,0],
         [208,47,0],
         [210,48,0],
         [211,49,0],
         [213,50,0],
         [214,51,0],
         [216,52,0],
         [217,53,0],
         [219,54,0],
         [221,55,0],
         [222,56,0],
         [224,57,0],
         [225,58,0],
         [227,59,0],
         [229,60,0],
         [230,61,0],
         [232,62,0],
         [233,63,0],
         [234,64,0],
         [235,66,0],
         [235,68,0],
         [235,70,0],
         [236,71,0],
         [236,73,0],
         [236,75,0],
         [237,77,0],
         [237,79,0],
         [237,80,0],
         [238,82,0],
         [238,84,0],
         [238,86,0],
         [239,87,0],
         [239,89,0],
         [239,91,0],
         [240,93,0],
         [240,95,0],
         [240,96,0],
         [241,98,0],
         [241,100,0],
         [241,102,0],
         [242,103,0],
         [242,105,0],
         [242,107,0],
         [243,109,0],
         [243,111,0],
         [243,112,0],
         [244,114,0],
         [244,116,0],
         [244,118,0],
         [245,119,0],
         [245,121,0],
         [246,123,0],
         [246,125,0],
         [246,127,0],
         [247,128,0],
         [247,130,0],
         [247,132,0],
         [248,134,0],
         [248,136,0],
         [248,137,0],
         [249,139,0],
         [249,141,0],
         [249,143,0],
         [250,144,0],
         [250,146,0],
         [250,148,0],
         [251,150,0],
         [251,152,0],
         [251,153,0],
         [252,155,0],
         [252,157,0],
         [252,159,0],
         [253,160,0],
         [253,162,0],
         [253,164,0],
         [254,166,0],
         [254,168,0],
         [254,169,0],
         [255,171,0],
         [255,173,0],
         [255,175,0],
         [255,177,0],
         [255,177,0],
         [255,178,0],
         [255,179,0],
         [255,179,0],
         [255,180,0],
         [255,180,0],
         [255,181,0],
         [255,181,0],
         [255,182,0],
         [255,183,0],
         [255,183,0],
         [255,184,0],
         [255,184,0],
         [255,185,0],
         [255,186,0],
         [255,186,0],
         [255,187,0],
         [255,187,0],
         [255,188,0],
         [255,188,0],
         [255,189,0],
         [255,190,0],
         [255,190,0],
         [255,191,0],
         [255,191,0],
         [255,192,0],
         [255,193,0],
         [255,193,0],
         [255,194,0],
         [255,194,0],
         [255,195,0],
         [255,195,0],
         [255,196,0],
         [255,197,0],
         [255,197,0],
         [255,198,0],
         [255,198,0],
         [255,199,0],
         [255,200,0],
         [255,200,0],
         [255,201,0],
         [255,201,0],
         [255,202,0],
         [255,202,0],
         [255,203,0],
         [255,204,0],
         [255,204,0],
         [255,205,0],
         [255,205,0],
         [255,206,0],
         [255,206,0],
         [255,207,0],
         [255,208,0],
         [255,208,0],
         [255,209,0],
         [255,209,0],
         [255,210,0],
         [255,211,0],
         [255,211,0],
         [255,212,0],
         [255,212,0],
         [255,213,0],
         [255,213,0],
         [255,214,0]
    ])
    
    cm = numpy.array([[255,128,0],
    [255,126,1],
    [254,125,2],
    [253,124,3],
    [252,123,4],
    [251,122,4],
    [250,121,5],
    [249,120,6],
    [248,119,7],
    [247,118,8],
    [246,117,9],
    [245,115,10],
    [244,114,11],
    [243,113,12],
    [242,112,13],
    [241,111,13],
    [240,110,14],
    [239,109,15],
    [239,108,16],
    [238,107,17],
    [237,106,18],
    [236,105,19],
    [235,104,20],
    [234,102,21],
    [233,101,22],
    [232,100,22],
    [231,99,23],
    [230,98,24],
    [229,97,25],
    [228,96,26],
    [227,95,27],
    [226,94,28],
    [225,93,29],
    [224,92,30],
    [223,90,30],
    [222,89,31],
    [221,88,32],
    [220,87,33],
    [219,86,34],
    [218,85,35],
    [217,84,36],
    [217,83,37],
    [216,82,38],
    [215,81,39],
    [214,80,39],
    [213,78,40],
    [212,77,41],
    [211,76,42],
    [210,75,43],
    [209,74,44],
    [208,73,45],
    [207,72,46],
    [206,71,47],
    [205,70,48],
    [204,69,48],
    [203,67,49],
    [202,66,50],
    [201,65,51],
    [200,64,52],
    [199,63,53],
    [198,62,54],
    [197,61,55],
    [196,60,56],
    [195,59,57],
    [194,58,57],
    [193,57,59],
    [192,56,60],
    [191,55,61],
    [190,54,62],
    [189,53,63],
    [188,52,64],
    [187,51,65],
    [186,50,66],
    [185,50,67],
    [184,49,68],
    [183,48,69],
    [182,47,71],
    [180,46,72],
    [179,45,73],
    [178,44,74],
    [177,43,75],
    [176,42,76],
    [175,41,77],
    [174,40,78],
    [173,40,79],
    [172,39,80],
    [171,38,81],
    [170,37,83],
    [169,36,84],
    [167,35,85],
    [166,34,86],
    [165,33,87],
    [164,32,88],
    [163,31,89],
    [162,30,90],
    [161,30,91],
    [160,29,92],
    [159,28,93],
    [158,27,95],
    [157,26,96],
    [156,25,97],
    [155,24,98],
    [153,23,99],
    [152,22,100],
    [151,21,101],
    [150,20,102],
    [149,20,103],
    [148,19,104],
    [147,18,105],
    [146,17,107],
    [145,16,108],
    [144,15,109],
    [143,14,110],
    [142,13,111],
    [141,12,112],
    [139,11,113],
    [138,10,114],
    [137,10,115],
    [136,9,116],
    [135,8,117],
    [134,7,119],
    [133,6,120],
    [132,5,121],
    [131,4,122],
    [130,3,123],
    [129,2,124],
    [128,1,125],
    [126,0,126],
    [125,0,127],
    [124,1,127],
    [123,2,128],
    [122,3,129],
    [121,4,129],
    [120,5,130],
    [119,6,130],
    [118,7,131],
    [117,8,131],
    [116,9,132],
    [115,10,132],
    [114,11,133],
    [113,12,133],
    [112,12,134],
    [111,13,134],
    [110,14,135],
    [109,15,135],
    [108,16,136],
    [107,17,136],
    [106,18,137],
    [105,19,137],
    [104,20,138],
    [103,21,138],
    [101,22,139],
    [100,23,139],
    [99,23,140],
    [98,24,140],
    [97,25,141],
    [96,26,141],
    [95,27,142],
    [94,28,142],
    [93,29,143],
    [92,30,143],
    [91,31,144],
    [90,32,144],
    [89,33,145],
    [88,34,145],
    [87,35,146],
    [86,35,146],
    [85,36,147],
    [84,37,147],
    [83,38,148],
    [82,39,148],
    [81,40,149],
    [80,41,149],
    [79,42,150],
    [77,43,150],
    [76,44,151],
    [75,45,151],
    [74,46,152],
    [73,47,152],
    [72,47,153],
    [71,48,153],
    [70,49,154],
    [69,50,154],
    [68,51,155],
    [67,52,155],
    [66,53,156],
    [65,54,157],
    [64,55,157],
    [63,56,158],
    [62,57,158],
    [61,58,159],
    [60,58,159],
    [59,59,159],
    [58,58,159],
    [57,58,158],
    [56,58,158],
    [55,58,157],
    [54,58,157],
    [53,57,157],
    [52,57,156],
    [51,57,156],
    [50,57,156],
    [49,57,155],
    [49,56,155],
    [48,56,155],
    [47,56,154],
    [46,56,154],
    [45,55,153],
    [44,55,153],
    [43,55,153],
    [42,55,152],
    [41,55,152],
    [40,54,152],
    [39,54,151],
    [38,54,151],
    [37,54,151],
    [36,54,150],
    [35,53,150],
    [35,53,149],
    [34,53,149],
    [33,53,149],
    [32,53,148],
    [31,52,148],
    [30,52,148],
    [29,52,147],
    [28,52,147],
    [27,52,147],
    [26,51,146],
    [25,51,146],
    [24,51,145],
    [23,51,145],
    [22,51,145],
    [21,50,144],
    [21,50,144],
    [20,50,144],
    [19,50,143],
    [18,50,143],
    [17,49,143],
    [16,49,142],
    [15,49,142],
    [14,49,141],
    [13,48,141],
    [12,48,141],
    [11,48,140],
    [10,48,140],
    [9,48,140],
    [8,47,139],
    [7,47,139],
    [7,47,139],
    [6,47,138],
    [5,47,138],
    [4,46,137],
    [3,46,137],
    [2,46,137],
    [1,46,136],
    [0,46,136]])[::-1]
    
    return cm


