class IdsManager:
    def __init__(self, maxid):
        self.maxid = maxid
        self.remandids = set()
        self.reset()

    def reset(self):
        self.remandids = set()
        for i in range(self.maxid):
            self.remandids.add(i)

    def get_new_id(self):
        if len(self.remandids) == 0:
            print("watchout: list of ids is empty")
            return -1
        
        return self.remandids.pop()

    def place_back(self, idx):
        if idx in self.remandids:
            return False
        
        self.remandids.add(idx)
        return True