import torch

class Model():
    def __init__(self, model, cfg):
        self.model = model
        self.lr    = cfg.lr
        self.optim = cfg.optim
        self.loss  = cfg.loss
    def train(self, data_path):
        pass
    def infer(selfm data_path):
        pass

if __name__ == '__main__':
	print("Hello Fruit Classification!")
