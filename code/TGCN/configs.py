import configparser


class Config:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        '''
        # training
        train_config = config['TRAIN']
        self.batch_size = int(train_config['BATCH_SIZE'])
        self.max_epochs = int(train_config['MAX_EPOCHS'])
        self.log_interval = int(train_config['LOG_INTERVAL'])
        self.num_samples = int(train_config['NUM_SAMPLES'])
        self.drop_p = float(train_config['DROP_P'])

        # optimizer
        opt_config = config['OPTIMIZER']
        self.init_lr = float(opt_config['INIT_LR'])
        self.adam_eps = float(opt_config['ADAM_EPS'])
        self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

        # GCN
        gcn_config = config['GCN']
        self.hidden_size = int(gcn_config['HIDDEN_SIZE'])
        self.num_stages = int(gcn_config['NUM_STAGES'])
        '''
        
        self.batch_size = 64
        self.max_epochs = 32
        self.log_interval = 1
        self.num_samples = 112
        self.drop_p = 0.3


        self.init_lr = 0.001
        self.adam_eps =  1e-3
        self.adam_weight_decay = 0

        self.hidden_size = 64
        self.num_stages = 20
        
        
        

    def __str__(self):
        return 'bs={}_ns={}_drop={}_lr={}_eps={}_wd={}'.format(
            self.batch_size, self.num_samples, self.drop_p, self.init_lr, self.adam_eps, self.adam_weight_decay
        )


if __name__ == '__main__':
    config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
    print(str(Config(config_path)))