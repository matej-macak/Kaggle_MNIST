from deep_mlp import train_mlp

transforms = {'scale':True, 'shear':False, 'rotate':True,
              'scale_interval':(0.9,1.1), 'rotate_interval':(-7.5,7.5),
              'shear_interval':(-0.05,0.05)}

early_stopping = {'patience':10000, 'patience_increase':2,
                  'improvement_threshold':0.995}    

train_mlp(dataset = "/Users/Matej/Dropbox/Programming/Python/Machine_Learning/MNIST/Data/train.csv",
         early_stopping = early_stopping, load_model_name = None,
         transforms = transforms)
         
