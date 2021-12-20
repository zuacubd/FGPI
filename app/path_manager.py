import sys
import os

class Datapath:

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_root_dir(self):
        '''
            returns the root directory of the data
        '''
        return self.data_dir

    def get_data_dir(self):
        '''
            returns the directory of the data
        '''
        return os.path.join(self.get_root_dir(), 'data')


    def get_data_raw_dir(self):
        '''
            returns the directory of the raw data
        '''
        return os.path.join(self.get_data_dir(), 'raw')


    def get_data_processed_dir(self):
        '''
            returns the directory of the processed data
        '''
        return os.path.join(self.get_data_dir(), 'processed')


    def get_features_dir(self):
        '''
            returns the directory of the features
        '''
        return os.path.join(self.get_root_dir(), 'features')


    def get_models_dir(self):
        '''
            returns the directory of the models
        '''
        return os.path.join(self.get_root_dir(), 'models')


    def get_results_dir(self):
        '''
            returns the directory of the results
        '''
        return os.path.join(self.get_root_dir(), 'results')


    def get_visualization_dir(self):
        '''
            returns the directory of the visualization
        '''
        return os.path.join(self.get_root_dir(), 'visualization')

