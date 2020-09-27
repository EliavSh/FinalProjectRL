import numpy as np
from runnable_scripts.Utils import get_config


class GameGrid:
    def __init__(self):
        """
        setting height and width of the game board - the only place with setting access
        :param height: int
        :param width: int
        """
        main_info = get_config("MainInfo")
        self.__height = int(main_info['board_height'])
        self.__width = int(main_info['board_width'])
        self.__states = np.reshape(np.array(range(self.__height * self.__width)), [self.__height, self.__width])  # every state has its matching int
        self.__values = np.zeros([self.__height, self.__width])

    def get_height(self):
        return self.__height

    def get_width(self):
        return self.__width

    def get_values(self):
        return self.__states
