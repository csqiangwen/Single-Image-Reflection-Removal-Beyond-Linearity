import os
import ntpath
import numpy as np
from PIL import Image
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.reflection_synthesis import ReflectionSynthesisModel
from util import util

opt = TestOptions().parse()
opt.batchSize = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = ReflectionSynthesisModel()
model.initialize(opt)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    img_path = model.get_image_paths()
    img_dir = os.path.join(opt.results_dir, '%s_%s_%s' % (opt.phase, opt.which_epoch, opt.type))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    print('%04d: process image... %s' % (i, img_path))
    for label, image_numpy in model.get_current_visuals_test().items():
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(img_dir, image_name)
        util.save_image(image_numpy, save_path)
        save_path = os.path.join(img_dir, '%s_W.npy' % name)
        np.save(save_path, model.W_A_reflection.cpu().numpy())
