import scipy.io
import tools

def visualize():
    ######
    result_num = '07'
    result_path = 'test_res/X_Y_pred_' + result_num + '_00000.mat'
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


if __name__ == '__main__':
    visualize()
