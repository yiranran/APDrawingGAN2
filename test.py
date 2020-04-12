import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch),reflesh=0, folder=opt.imagefolder)
    if opt.test_continuity_loss:
        file_name = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'continuity.txt')
        file_name1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch), 'continuity-r.txt')
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(file_name1):
            os.remove(file_name1)
    # test
    #model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.how_many:#test code only supports batch_size = 1, how_many means how many test images to run
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()#in test the loadSize is set to the same as fineSize
        img_path = model.get_image_paths()
        #if i % 5 == 0:
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
    if opt.model == 'regressor':
        print(model.cnt)
        print(model.value/model.cnt)
        print(model.minval)
        print(model.avg/model.cnt)
        print(model.max)
        html = os.path.join(web_dir,'cindex'+opt.imagefolder[6:]+'.html')
        f=open(html,'w')
        print('<table border="1" style=\"text-align:center;\">',file=f,end='')
        print('<tr>',file=f,end='')
        print('<td>image name</td>',file=f,end='')
        print('<td>realA</td>',file=f,end='')
        print('<td>realB</td>',file=f,end='')
        print('<td>fakeB</td>',file=f,end='')
        print('</tr>',file=f,end='')
        for info in model.info:
            basen = os.path.basename(info[0])[:-4]
            print('<tr>',file=f,end='')
            print('<td>%s</td>'%basen,file=f,end='')
            print('<td><img src=\"%s/%s_real_A.png\" style=\"width:44px\"></td>'%(opt.imagefolder,basen),file=f,end='')
            print('<td>%.4f</td>'%info[1],file=f,end='')
            print('<td>%.4f</td>'%info[2],file=f,end='')
            print('</tr>',file=f,end='')
        print('</table>',file=f,end='')
        f.close()
