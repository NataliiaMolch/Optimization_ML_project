# from concurrent.futures import ThreadPoolExecutor

import time
# from Monitor import Monitor
from HyperBand import HyperBand


"""
Run search in a separate thread, while measuring performance -- part that doesn't work
"""
# with ThreadPoolExecutor() as executor:
#     monitor = Monitor(10)
#     mem_thread = executor.submit(monitor.start_measurements(HPB))
#     print('harshdeep 2')
#     executor.submit(HPB.search())
#     monitor.continue_measuring = False
#     monitor.plot_results()

HPB = HyperBand(max_epoch=3)
HPB.search()
