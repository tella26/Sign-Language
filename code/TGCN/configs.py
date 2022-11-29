import configparser


class Config:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.batch_size = 4
        self.max_epochs = 2
        self.log_interval = 1
        self.num_samples = 222
        self.drop_p = 0.4


        self.init_lr = 0.001
        self.adam_eps =  1e-3
        self.adam_weight_decay = 0.1

        self.hidden_size = 64
        self.num_stages = 20
        
        
        

    def __str__(self):
        return 'bs={}_ns={}_drop={}_lr={}_eps={}_wd={}'.format(
            self.batch_size, self.num_samples, self.drop_p, self.init_lr, self.adam_eps, self.adam_weight_decay
        )


if __name__ == '__main__':
    config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
    print(str(Config(config_path)))