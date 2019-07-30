import os
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.reflection_removal import ReflectionRemovalModel
from util import util
import numpy as np
import ntpath

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = ReflectionRemovalModel()
model.initialize(opt)
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    img_path = model.get_image_paths()
    img_dir = os.path.join(opt.results_dir, opt.which_type, '%s_%s' % (opt.phase, opt.which_epoch))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]
    print('%04d: process image... %s' % (i, img_path))

    for label, image_numpy in model.get_current_visuals_test().items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(img_dir, image_name)
        util.save_image(image_numpy, save_path)