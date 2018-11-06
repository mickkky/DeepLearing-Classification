# -*- coding: UTF-8 -*-
import Augmentor
# 1. set dataset directory
p = Augmentor.Pipeline("directory/to/your/dataset")

# 2. augment options
# Rotate
p.rotate(probability=0.7,max_left_rotation=25, max_right_rotation=25)
# zoom probability 0.3
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
# resize
# p.resize(probability=1,height=224,width=224)
# 3. Set the  number of images after augment
p.sample(1500)
