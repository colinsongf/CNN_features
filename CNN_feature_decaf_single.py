__author__ = 'chensi'
import numpy as np
import sys
caffe_root = '/data/sichen/caffe_svcl/caffe_latest/'
sys.path.insert(0,caffe_root + 'python')
import caffe
import glob
import cPickle
from optparse import OptionParser
import time
import scipy.io as sio
import os.path
import os
from scipy.sparse import csr_matrix
caffe.set_mode_gpu()

def initial_network_custom(proto_path, model_path, mean_path):

    net = caffe.Classifier(proto_path,
                           model_path,
                           mean = np.load(caffe_root+mean_path),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def initial_network_vgg_center():

    net = caffe.Classifier(caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy_center.prototxt',
                           caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def initial_network_places_center():

    net = caffe.Classifier(caffe_root+'models/places/places205CNN_deploy.prototxt',
                           caffe_root+'models/places/places205CNN_iter_300000.caffemodel',
                           mean = np.load(caffe_root+'models/places/places_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net

def initial_network_vgg_ten_crops():

    net = caffe.Classifier(caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt',
                           caffe_root+'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def initial_network_alex_center():

    net = caffe.Classifier(caffe_root+'models/bvlc_reference_caffenet/deploy_center.prototxt',
                           caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def initial_network_alex_ten_crops():

    net = caffe.Classifier(caffe_root+'models/bvlc_reference_caffenet/deploy.prototxt',
                           caffe_root+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                           mean = np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def initial_network_finetune_SUN():

    net = caffe.Classifier('/data/sichen/caffe_svcl/caffe_googlenet/models/googlenet_places/SUN_finetune_deploy.prototxt',
                           '/data/sichen/caffe_svcl/caffe_googlenet/models/googlenet_places/SUN_places_net_397_iter_25000.caffemodel',
                           mean = np.load(caffe_root+'models/places/places_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256,256))
    return net
def get_options_parser():
    parser = OptionParser()
    parser.add_option('-i','--input_path',dest='img_input_path')
    parser.add_option('-o','--output',dest='feature_output_path',default=None)
    parser.add_option('--layer',dest='layer',default='conv5')
    parser.add_option('--mode',dest='mode',default='custom')
    parser.add_option('--mean',dest='mean_path',default='python/caffe/imagenet/ilsvrc_2012_mean.npy')
    parser.add_option('--prototxt', dest = 'prototxt_path')
    parser.add_option('--model', dest = 'model')
    parser.add_option('--center', dest = 'center', default = True);
    return parser



def main():
    parser = get_options_parser()

    (options, args) = parser.parse_args()

    out_path = options.feature_output_path
    if not out_path:
	out_path = options.img_input_path
    file_name, fileExtension = os.path.splitext(options.img_input_path)

    layer = options.layer
    oversample = False;
    
    if options.mode == 'AlexNetCenter' :
        net = initial_network_alex_center()
    elif options.mode == 'VggNetCenter' : 
        net = initial_network_vgg_center()
    elif options.mode == 'places' :
        net = initial_network_places_center()
    elif options.mode == 'AlexNetTen' :
        net = initial_network_alex_ten_crops()
        oversample = True
    elif options.mode == 'VggNetTen' : 
        net = initial_network_vgg_ten_crops()
        oversample = True
    elif options.mode == 'SUN':
    	net = initial_network_finetune_SUN()
    	oversample = False
    else :
        net = initial_network_custom(options.prototxt_path, options.model, options.mean_path)
        oversample = not options.center
        

    start_time = time.time()
    print 'extracting the CNN feature, layer: %s' %(layer)
    net.predict([caffe.io.load_image(options.img_input_path)],oversample)        
    feature_temp = net.blobs[layer].data
    out_path1 = out_path+os.path.basename(options.img_input_path).replace(fileExtension,'_CNN_'+layer+'_feature'+'.mat')
    try:
        sio.savemat(out_path1,{'CNN_feature':feature_temp})
    except:
        os.makedirs(os.path.dirname(out_path1))
        sio.savemat(out_path1,{'CNN_feature':feature_temp})

    print 'time used:'+str(time.time()-start_time)+'s'





if __name__ == '__main__':
    main()





