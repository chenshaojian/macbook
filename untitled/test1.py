#!/usr/bin/python
# coding:utf-8
import threading
import datetime
import logging
import time
import random
import requests

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s', )
list = ['192.168.0.0', '192.168.0.1', '192.168.0.2', '192.168.0.3', '192.168.0.4', '192.168.0.5', '192.168.0.6',
        '192.168.0.7', '192.168.0.8', '192.168.0.9',
        '192.168.0.10', '192.168.0.11', '192.168.0.12', '192.168.0.13', '192.168.0.14', '192.168.0.15', '192.168.0.16',
        '192.168.0.17', '192.168.0.18']
urls = [
    'http://www.baidu.com',
    'http://www.amazon.com',
    'http://www.ebay.com',
    'http://www.alibaba.com',
    'http://www.sina.com'
]


class Test(threading.Thread):
    def __init__(self, threadingSum,shop_id,min_time,max_time,date):
        threading.Thread.__init__(self)
        self.shop_id = shop_id
        self.min_time = min_time
        self.max_time = max_time
        self.date = date
        self.threadingSum = threadingSum

    def run(self):
        with self.threadingSum:
            get_status(self.shop_id, self.min_time, self.max_time, self.date)
            time.sleep(1)

            #print r.status_code,self.url
            logging.debug("%s start!" % self.shop_id)
            #time.sleep(random.randint(1, 3))
            #logging.debug('%s Done!' % self.ip)
def get_status(shop_id,min_time,max_time,date):
    print shop_id,min_time,max_time,date

if __name__ == "__main__":
    # 设置线程数
    threadingSum = threading.Semaphore(2)

    # 启动线程
    for url in urls:
        t = Test(threadingSum,url,url,url,url)
        t.setDaemon(True)
        t.start()
        # 等待所有线程结束
    for t in threading.enumerate():
        if t is threading.currentThread():
            continue
        t.join()

    logging.debug('Done!')
