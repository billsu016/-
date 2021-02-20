# python类的继承
# example 1 -- 未向实例c中传参

class Person(object):  # 定义一个父类

    def talk(self):  # 父类中的方法
        print('person is talking...')

class Chinese(Person):  # 定义一个子类，继承父类

    def walk(self):  # 定义子类本身的方法
        print('is walking...')

c = Chinese()
c.talk()  # 调用父类的方法
c.walk()  # 调用子类的方法

# example 2 -- 向实例c中传参

class Person(object):

    def __init__(self, name, age):  # 定义父类的构造方法
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):
        print('person is talking...')

class Chinese(Person):

    def __init__(self, name, age, language):  # 先继承，再重构
        Person.__init__(self, name, age)
        # super(Chinese, self).__init__(name, age)
        # %33、%34均能继承父类的构造方法
        self.language = language
        print(self.name, self.age, self.weight, self.language)

    def talk(self):  # 子类重构方法
        print('%s is speaking Chinese' % self.name)

    def walk(self):
        print('is walking...')

class American(Person):
    pass

c = Chinese('yangyang', 21, 'Chinese')
c.talk()

# %48 实例化对象c，并调用子类-->%32 调用子类__init__()，并传入参数，
# c传入self，'yangyang'传入name，21传入age-->%33 调用父类__init__()，并传入参数
