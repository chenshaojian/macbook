#!/usr/bin/python
import numpy as np
import json


DIRECTION_IN = 1
DIRECTION_OUT = -1

num=0
class PedestrianOpticalFlow(object):

    def __init__(self):
        self.tms = 0
        self.num_pt = 0
        self.direction = DIRECTION_IN



class PedestrianCounter(object):


    def __init__(self):

        self.fps =24
        self.dt = 2


        self.num_people_in = 0
        self.num_people_out = 0

        self.num_batch_out = 0
        self.num_batch_in = 0


        self.num_slow_people_in = 0
        self.num_slow_people_out = 0

        self.num_slow_batch_in = 0
        self.num_slow_batch_out = 0


    def get_pedestrian_optical_flow(self, js):

        assert isinstance(js, dict)
        data = js['data']
        in_pts = []
        out_pts = []
        vs = []
        for item in data:
            x = item['x']
            y = item['y']
            t = item['t']
            v = item['v']
            if abs(y-200)<30:
                if v<0:
                    in_pts.append(t)
                elif v>0:
                    out_pts.append(t)
            vs.append(abs(v))

        v_sum = np.sum(vs)
        bins = range(0,1500, int(self.fps))
        in_hists,edges = np.histogram(in_pts, bins)
        out_hists,edges = np.histogram(out_pts, bins)

        #cls.num_people_in = len(np.where(in_hists>10))
        #cls.num_people_out = len(np.where(out_hists>10))
        thres = 40
        num_in = np.where(in_hists>thres)
        num_out = np.where(out_hists>thres)
        num_in = num_in[0]
        num_out = num_out[0]

        batch_in = np.abs(num_in[0:-1]-num_in[1:])

        batch_out = np.abs(num_out[0:-1]-num_out[1:])
        #print in_hists
        #print num_in, num_out
        #print batch_in, batch_out

        self.num_people_in = len(num_in)

        self.num_people_out = len(num_out)
        #print self.num_people_in, self.num_batch_in, self.num_people_out, self.num_batch_out
        self.num_people_in = min([len(np.where(batch_in>3)[0])+1, self.num_people_in])
        self.num_people_out = min([len(np.where(batch_out>3)[0])+1,self.num_people_out])
        bi2 = np.abs(batch_in[:-1]-batch_in[1:])
        bo2 = np.abs(batch_out[:-1]-batch_out[1:])
        #print in_hists, out_hists
        #print num_in,num_out
        #print batch_in, batch_out
        #print bi2, bo2
        self.num_batch_in = min([len(np.where(bi2>3)[0])+1, self.num_people_in])
        self.num_batch_out = min([len(np.where(bo2>3)[0])+1, self.num_people_out])
        k = 100
        r = np.random.random_integers(k)
        v_sum=r
        if r<15:
            self.num_slow_batch_in = self.num_batch_in
            self.num_slow_people_in = self.num_people_in
        self.print_values()
        self.sum_all()
    def print_values(self):
        global num
        for key in [key for key in dir(self) if not key.startswith('__')]:
            if callable(getattr(self, key)):
                continue
            else:
                #print key, getattr(self, key)
                if key=='num_slow_batch_in':
                    num +=getattr(self, key)



    def sum_all(self):
        global dic_all
        dic_all['num_people_in']+=self.num_people_in
        dic_all['num_people_out']+=self.num_people_out
        dic_all['num_batch_out']+=self.num_batch_out
        dic_all['num_batch_in']+=self.num_batch_in
        dic_all['num_slow_people_in']+=self.num_slow_people_in
        dic_all['num_slow_people_out']+=self.num_slow_people_out
        dic_all['num_slow_batch_in']+=self.num_slow_batch_in
        dic_all['num_slow_batch_out']+=self.num_slow_batch_out



if __name__=='__main__':
    dic_all = {'num_people_in': 0, 'num_people_out': 0, 'num_batch_out': 0, 'num_batch_in': 0, \
               'num_slow_people_in': 0, 'num_slow_people_out': 0, 'num_slow_batch_in': 0, 'num_slow_batch_out': 0
               }
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-09')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p=PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    '''
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-10')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-11')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-12')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-13')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-14')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-15')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-16')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-17')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    f = open('/Users/chenshj/Desktop/camtest/555-20160803-18')
    s = f.readlines()
    for line in s:
        js = json.loads(''.join(line))
        p = PedestrianCounter()
        p.get_pedestrian_optical_flow(js)
    '''
    print dic_all
    print num









