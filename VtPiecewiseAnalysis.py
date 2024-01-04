import pandas as pd
import numpy as np
import pwlf,  piecewise_regression
import matplotlib as plt
import matplotlib.pyplot as plto
from tabulate import tabulate
import plotly.express as plex
import os 
import re

i_VelData = 0
j_VelData =0
pattern1 = r'^\d\.\d\d$'
pattern2 = r'\d\d?'
Veldataframe = pd.DataFrame(index=range(200), columns=range(200))


granddirectory = "E:\Home\Sharif\Red blood cell\Data Analysis\GrandResults"
#Generate File names using Directories
directory = os.getcwd() # get current working directory
for entry1 in os.listdir(directory): # loop over all entries
    if os.path.isdir(os.path.join(directory, entry1)): # check if entry is a directory
        if re.match(pattern1, entry1): # check if entry matches the pattern
            directory1 = os.path.join(directory, entry1)
            # Veldataframe.iat[i_VelData,0] = directory1       #lets forget it for now, new way of tablinnn
            # i_VelData += 1 
            print("________NeW FIle___")
            print(entry1)
            print("_____")
            for entry2 in os.listdir(directory1):
                 if os.path.isdir(os.path.join(directory1, entry2)):
                     if re.match(pattern2, entry2):
                         print(entry2)
                         directory2 = os.path.join(directory1, entry2)
                         Veldataframe.iat[j_VelData,0 ] = directory2[57:]
                         suffix = directory2[57:].replace("\\","_")
                         directoryresuslts = "Results_"+suffix
                         path1 = os.path.join(directory2, directoryresuslts) 
                         path2 = os.path.join(granddirectory, directoryresuslts) 
                         if os.path.isdir(path1) == False:
                            os.mkdir(path1)     # One file in the same folder the data is
                         if os.path.isdir(path2) == False:    
                            os.mkdir(path2)     # One file in the mother folder all data are    
                         
                         #Constants and Arrays
                         beta = 1.3795e-7
                         k = 9.0594e-5
                         MaxInterval = 8
                         MaxBreakpoints = 3                       # The real number of breakpoints is MAxbreakpoints+1
                         BICsList = np.empty((MaxBreakpoints+1,MaxInterval))    # Array to save BIC
                         alphasList = np.zeros((3,MaxInterval))                     #Array to save the slopes
                         breakpointsList = np.empty((MaxBreakpoints+1,MaxInterval))    # Array to save Breakpoints

                        #Read Data
                         d = pd.read_table(directory2+"\\StretchingA 0unaveraged.txt", names=["v_x", "v_y", "v_z", "v_piezo"])
                         dataSize = len(d)
                         lastT = dataSize*0.002 +0.002
                         d["time"] = np.arange(0.002, lastT, 0.002)
                         d["F"] = d["v_y"] * k * beta * 1e12
                         d["absvpiezo"] = abs(d["v_piezo"])
                         d = d.reset_index()
                         print("________data size_______ = "+ str(lastT))
                         newT = d.loc[0:10000, "time"].to_numpy()
                         absvpiezodata = d.loc[0:10000, "absvpiezo"].to_numpy()

                         try:
                            # V_t fit with one break point
                            pwvt1_fit = piecewise_regression.Fit(newT, absvpiezodata,  n_breakpoints=1, n_boot =10)
                            pwvt1_fit.plot_data(color="grey", s=1)
                            pwvt1_fit.plot_fit(color="red", linewidth=1)
                            pwvt1_fit.plot_breakpoints() 
                            pwvt1_fit.plot_breakpoint_confidence_intervals()
                            plto.xlabel("x")
                            plto.ylabel("y")
                            plto.savefig(path1+"\\Vfit1.png")
                            plto.savefig(path2+"\\Vfit1.png")
                            plto.close()

                            #V-t fit with 2 breakpints


                            # pwvt2_fit = piecewise_regression.Fit(newT, absvpiezodata,  n_breakpoints=2, n_boot =200)


                            # pwvt2_fit.plot_data(color="grey", s=1)
                            # pwvt2_fit.plot_fit(color="red", linewidth=1)
                            # pwvt2_fit.plot_breakpoints()
                            # pwvt2_fit.plot_breakpoint_confidence_intervals()
                            # plto.xlabel("x")
                            # plto.ylabel("y")
                            # plto.savefig(path1+"\\Vfit2.png")
                            # plto.savefig(path2+"\\Vfit2.png")
                            # plto.close()


                            # pwvt1_results = pwvt1_fit.get_results()
                            # vt1Bic = pwvt1_results["bic"]
                            # pwvt2_results = pwvt2_fit.get_results()
                            # vt2Bic = pwvt2_results["bic"]
                            # if vt1Bic<=vt2Bic:
                            alpha_hat1 = pwvt1_fit.best_muggeo.best_fit.raw_params[1]
                            VtSlope = 2.5 * (alpha_hat1)
                            # else:
                            #     alpha_hat2 = pwvt2_fit.best_muggeo.best_fit.raw_params[1]
                            #     beta_hats2 = pwvt2_fit.best_muggeo.best_fit.raw_params[2:4]
                            #     VtSlope = 2.5 * (alpha_hat2 + beta_hats2[1])

                            Veldataframe.iat[j_VelData,1] = VtSlope
                            j_VelData +=1
                         except:
                            Veldataframe.iat[j_VelData,1] = 0
                            j_VelData +=1



Veldataframe.to_csv(directory + f"\\Velocities_{granddirectory[57:]}.csv", header=False, index=False)                            
Veldataframe.to_csv(granddirectory+f"\\Velocities_{granddirectory[57:]}.csv", header=False, index=False)                                    
                             
                             
                             






                

