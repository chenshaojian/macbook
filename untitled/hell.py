import threadpool
import time
import requests
import time
from threading import Lock,Thread
import threading
lock = Lock()
import urllib2

urls = [
    'http://www.google.com',
    'http://www.amazon.com',
    'http://www.ebay.com',
    'http://www.alibaba.com',
    'http://www.reddit.com'
]

def myRequest(url):
    try:
        print '12'
        r = requests.get(url)
        print url, r.status_code
    except:
        pass
    # print  r.text




def timeCost(request, n):
  print "Elapsed time: %s" % (time.time()-start)

start = time.time()
pool = threadpool.ThreadPool(8)
reqs = threadpool.makeRequests(myRequest, urls, timeCost)
[ pool.putRequest(req) for req in reqs ]
#print "current has %d threads" % (threading.activeCount() - 1)
pool.wait()