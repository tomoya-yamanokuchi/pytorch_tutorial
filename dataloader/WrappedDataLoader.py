'''
チュートリアルに従って作ったもの
'''

class WrappedDataLoader:
    def __init__(self, dl, dev=None):
        self.dl   = dl
        self.dev  = dev
        self.func = self.preprocess

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


    def preprocess(self, x, y):
        if self.dev is None: return x.view(-1, 1, 28, 28), y
        else               : return x.view(-1, 1, 28, 28).to(self.dev), y.to(self.dev)