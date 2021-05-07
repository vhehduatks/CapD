class test2:
    def __init__(self,opt):
      self.name = opt.name
      self.age = opt.age
      temp=1
      if temp:
          print("work",self.name,self.age)

    def func(self):
        print(self.name)
        temp=2
        print(temp)
        self._hidden()
        print('hidden2',self._hidden2(temp))

    def _hidden(self):
        print("hidden func")

    def _hidden2(self,temp):
        return temp+1
