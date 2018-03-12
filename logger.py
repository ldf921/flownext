import datetime
import os

class FileLog:
    __timezone__ = 8

    def __init__(self, path, name=None):
        self.f = open(path, 'w')

    @classmethod
    def _localtime(cls):
        return datetime.datetime.utcnow() + datetime.timedelta(hours=cls.__timezone__)

    def _timestamp(self):
        return (self._localtime()).strftime('[%Y/%m/%d %H:%M:%S] ')
    
    def log(self, msg, end='\n'):
        self.f.write(self._timestamp() + msg + end)
        self.f.flush()