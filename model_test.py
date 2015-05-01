import longmantide as tide
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

model = tide.TideModel()
model.start_time = datetime(2015,4,23,0,0,0)
model.increment = 60*10 # Run every 10 minutes
model.latitude = 40.7914
model.longitude = 282.1414
model.altitude = 370.
model.duration = 7
model.run_model()
model.write('output.txt')
model.plot()


#net_results = np.loadtxt('net_results.txt',skiprows=10,usecols=[2])
#net_time = np.arange(len(net_results))
#plt.plot(net_time,net_results,color='r')

#gm,gs,g,d = run_model(t0,600,7)
#plt.plot_date(model.model_times,model.gravity,'-k',linewidth=2)
#plt.show()
