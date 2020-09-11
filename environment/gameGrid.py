import numpy as np


class GameGrid:
    def __init__(self, height, width):
        """
        setting values_per_column - the only place with setting access
        :param height: int
        :param width: int
        """
        self.__height = height
        self.__width = width
        self.__states = np.reshape(np.array(range(height * width)), [height, width])  # every state has its matching int
        self.__values = np.zeros([height, width])

    def get_height(self):
        return self.__height

    def get_width(self):
        return self.__width

    def get_values(self):
        return self.__states
