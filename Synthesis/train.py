import time
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.reflection_synthesis import ReflectionSynthesisModel
import os
from PIL import Image
from util import util

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = ReflectionSynthesisModel()
model.initialize(opt)
total_steps = 0

def print_current_errors(epoch, i, errors, t, t_data, log_name):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

for epoch in range(opt.epoch_count, opt.niter + 50 + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            img_dir = os.path.join(opt.checkpoints_dir, 'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for label, image_numpy in model.get_current_visuals_train().items():
                img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
            print_current_errors(epoch, epoch_iter, errors, t, t_data, log_name)
            
        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
