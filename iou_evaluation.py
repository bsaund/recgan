import os
import numpy as np
import scipy.io
import tensorflow as tf
import tools
import IPython
import random

GPU0 = '0'


def get_iou(arr1, arr2):
    """Returns the intersections over union for two numerical arrays, with a threshold of 0.5"""
    inter = np.logical_and(arr1 > 0.5, arr2 > 0.5)
    union = np.logical_or(arr1 > 0.5, arr2 > 0.5)

    return float(np.sum(inter)) / np.sum(union)


class Model:
    def __init__(self):
        model_path = './Model_released/'
        # model_path = '../trial_02/train_mod/'
        if not os.path.isfile(model_path + 'model.cptk.data-00000-of-00001'):
            print ('please download our released model first!')
            return

    
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0
        # with tf.Session(config=config) as sess:
        self.sess = tf.Session(config=config)
        saver = tf.train.import_meta_graph( model_path +'model.cptk.meta', clear_devices=True)
        saver.restore(self.sess, model_path+ 'model.cptk')
        print ('model restored!')
        self.X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        self.Y_pred = tf.get_default_graph().get_tensor_by_name("aeu/Sigmoid:0")
        self.Y_pred_64 = tf.get_default_graph().get_tensor_by_name("aeu/conv3d_transpose_3:0")
        self.Y_pred_128 = tf.get_default_graph().get_tensor_by_name("aeu/conv3d_transpose_4:0")

    
        
        

    def iou_single_eval(self, x_path, y_true_path):
        ####### load sample data
        x_sample = tools.Data.load_single_voxel_grid(x_path, out_vox_res=64)
        y_true = tools.Data.load_single_voxel_grid(y_true_path, out_vox_res=256)

        ####### load model + testing
        # config = tf.ConfigProto(allow_soft_placement=True)
        # with tf.Session(config=config) as sess:
        x_sample = x_sample.reshape(1, 64, 64, 64, 1)
        # y_pred_256, y_pred_64, y_pred_128 = self.sess.run([self.Y_pred,
        #                                                    self.Y_pred_64,
        #                                                    self.Y_pred_128],
        #                                                   feed_dict={self.X: x_sample})
        y_pred = self.sess.run([self.Y_pred], feed_dict={self.X: x_sample})
        iou = get(y_pred_256, y_true)
        # iou_256 = get_iou(y_pred_256, y_true)
        # IPython.embed()
        # iou_128 = get_iou(y_pred_128, y_true[::2, ::2, ::2, :])
        # iou_64 = get_iou(y_pred_64, y_true[::4, ::4, ::4, :])
        
        # print "iou_256", iou_256
        # print "iou_128", iou_128
        # print "iou_64", iou_64
        return iou


        ###### save result
        # x_sample = x_sample.reshape(64, 64, 64)
        # y_pred = y_pred.reshape(256, 256, 256)
        # x_sample = x_sample.astype(np.int8)
        # y_pred = y_pred.astype(np.float16)
        # y_true = y_true.astype(np.int8)
        # to_save = {'X_test': x_sample, 'Y_test_pred': y_pred, 'Y_test_true': y_true}
        # scipy.io.savemat('demo_result.mat', to_save, do_compression=True)
        # print ('results saved.')


    def iou_eval(self, cat):
        # x_path = './Data_sample/P1_03001627_chair/test_25d_vox256/1c08f2aa305f124262e682c9809bff14_0_0_0.npz'
        # y_true_path = './Data_sample/P1_03001627_chair/test_3d_vox256/1c08f2aa305f124262e682c9809bff14_0_0_0.npz'

        # cat = "P1_03001627_chair"
        views = []
        for fileparts in os.walk('./Data_sample/' + cat + '/test_3d_vox256'):
            views = fileparts[2]

        ious = []

        for i in range(50):
            item = random.choice(views)
            x_path = "./Data_sample/" + cat + "/test_25d_vox256/" + item
            y_true_path = "./Data_sample/" + cat + "/test_3d_vox256/" + item
        
            ious.append(self.iou_single_eval(x_path, y_true_path))

        print "Results for", cat, 
        print ": iou mean: ", np.mean(ious), u'\u00B1', np.std(ious)
        # IPython.embed()
        # IPython.embed()

    def iou_eval_all_cat(self):
        cats = ['P1_02828884_bench','P1_03001627_chair','P1_04256520_coach', 'P1_04379243_table']

        for cat in cats:
            self.iou_eval(cat)

    
def visualize():
    ######
    result_path = 'demo_result.mat'
    mat = scipy.io.loadmat(result_path)
    x_sample = mat['X_test']
    y_pred = mat['Y_test_pred']
    y_true = mat['Y_test_true']

    ######  if the GPU serve is able to visualize, otherwise comment the following lines
    th = 0.5
    y_pred[y_pred >= th] = 1
    y_pred[y_pred < th] = 0
    tools.Data.plotFromVoxels(x_sample, title='x_sample')
    tools.Data.plotFromVoxels(y_pred, title='y_pred')
    tools.Data.plotFromVoxels(y_true, title='y_true')
    from matplotlib.pyplot import show
    show()

#########################
if __name__ == '__main__':
    m = Model()
    m.iou_eval_all_cat()


