class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sceneflow':
            return './dataset/SceneFlow/'
        elif dataset == 'kitti15':
            return './dataset/kitti2015/training/'
        elif dataset == 'kitti12':
            return './dataset/kitti2012/training/'
        elif dataset == 'middlebury':
            return './dataset/MiddEval3/trainingH/'
        elif dataset == 'apolloscape':
            return './dataset/apolloscape/stereo_train/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
