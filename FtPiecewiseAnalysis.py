import pandas as pd
import numpy as np
import piecewise_regression
import matplotlib.pyplot as plto
import os
from concurrent.futures import ThreadPoolExecutor

def process_file(filename, k, beta):
    parent_dir = "./RBC data_all/" + filename.replace("\\","/")
    suffix = parent_dir[15:].replace("/","_")
    directory = "Results_"+suffix
    realparent = "./GrandResults"
    path1 = os.path.join(parent_dir, directory) 
    path2 = os.path.join(realparent, directory) 
    if os.path.isdir(path1) == False:
        os.mkdir(path1)     # One file in the same folder the data is
    if os.path.isdir(path2) == False:    
        os.mkdir(path2)     # One file in the mother folder all data are
    
    
    #Constants and arrays to save data
    
    beta = beta
    k = k
    MaxInterval = 8
    MaxBreakpoints = 4                     # The real number of breakpoints is MAxbreakpoints-1
    BICsList = np.zeros((MaxBreakpoints+1,MaxInterval))    # Array to save BIC
    
    alphasList0 =np.zeros((1,MaxInterval))
    
    alphasList1 = np.zeros((2,MaxInterval))                     #Array to save the slopes
    breakpointsList1 = np.zeros((2,MaxInterval))    # Array to save Breakpoints
    
    alphasList2 = np.zeros((3,MaxInterval))                     #Array to save the slopes
    breakpointsList2 = np.zeros((3,MaxInterval))    # Array to save Breakpoints
    
    alphasList3 = np.zeros((4,MaxInterval))                     #Array to save the slopes
    breakpointsList3 = np.zeros((4,MaxInterval))    # Array to save Breakpoints
    
    # Reading Data and storing the columns of data frame into arrays
    print("opened_: "+filename)
    d = pd.read_table(parent_dir + "/StretchingA 0unaveraged.txt", names=["v_x", "v_y", "v_z", "v_piezo"])
    dataSize = len(d)
    lastT = dataSize*0.0002 
    d["time"] = np.arange(0, lastT, 0.0002)
    d["F"] = d["v_y"] * k * beta * 1e12
    d["absvpiezo"] = abs(d["v_piezo"])
    d = d.reset_index()
    print("________data size_______ = "+ str(lastT))
    newT = d.loc[0:10000, "time"].to_numpy()
    absvpiezodata = d.loc[0:10000, "absvpiezo"].to_numpy()
    
    # Plotting V_piezo to time
    #Finding xmax by 1 point fit to V-t
    
    pwvt1_fit = piecewise_regression.Fit(newT, absvpiezodata,  n_breakpoints=1, n_boot =40)
    pwvt1_fit.plot_data(color="grey", s=1)
    pwvt1_fit.plot_fit(color="red", linewidth=1)
    pwvt1_fit.plot_breakpoints()
    pwvt1_fit.plot_breakpoint_confidence_intervals()
    plto.xlabel("t(s)")
    plto.ylabel("V_piezo")
    plto.savefig(path1+"/Vfit1.png", dpi=300)
    plto.close()
    
    pw_results = pwvt1_fit.get_results()
    
    Vtbreak = pw_results["estimates"]["breakpoint1"]["estimate"]
    print(Vtbreak)
    xmax = int(np.floor(Vtbreak*5000))
    print("_______"+str(xmax))
    
    # ____Fitting
    
    for i in range(MaxInterval):
            
        tdata = np.log(abs(d.loc[xmax:xmax+((i+1)*5000), "time"]).to_numpy())
        Ydata = np.log(abs(d.loc[xmax:xmax+((i+1)*5000), "F"]).to_numpy())
        for j in range(MaxBreakpoints):    
            try:
                # Given some data, fit the model
                if j==0:
                    ms = piecewise_regression.ModelSelection(tdata , Ydata, max_breakpoints=0)
                    BICsList[j][i] = ms.model_summaries[0]["bic"] 
                    plto.scatter(tdata, Ydata, color="grey", s=1)
                    intercept0 = ms.model_summaries[0]["estimates"]["const"]
                    slope0 = ms.model_summaries[0]["estimates"]["alpha1"]
                    xx_plot = np.linspace(min(tdata), max(tdata), 100)
                    yy_plot = intercept0 + slope0 * xx_plot
                    plto.plot(xx_plot, yy_plot, color="red", linewidth=1)
                    plto.xlabel("log(t)")
                    plto.ylabel("log(F)")
                    plto.title(str((i+1)*500)+"BrPo"+str(j))
                    plto.savefig(path1+"/figIndex_"+str((i+1)*500)+"BrPo"+str(j)+".png", dpi=300)
                    plto.close()
                    alphasList0[0][i] = slope0
    
                else:
                
                    pw_fit = piecewise_regression.Fit(tdata, Ydata,  n_breakpoints=(j), n_boot =3000)
    
                    # Print a summary of the fit
                    pw_fit.summary()
                    rss = pw_fit.best_muggeo.best_fit.residual_sum_squares
                    n_obs = len(tdata)
                    k = 2  # No. parameters
                    pw_results = pw_fit.get_results()
                    BICsList[j][i] = pw_results["bic"]
    
    
                    # Plot the data, fit, breakpoints and confidence intervals
                    pw_fit.plot_data(color="grey", s=1)
                    # Pass in standard matplotlib keywords to control any of the plots
                    pw_fit.plot_fit(color="red", linewidth=1)
                    pw_fit.plot_breakpoints()
                    pw_fit.plot_breakpoint_confidence_intervals()
                    plto.xlabel("log(t)")
                    plto.ylabel("log(F)")
                    plto.title(str((i+1)*500)+"BrPo"+str(j))
                    plto.savefig(path1+"/figIndex_"+str((i+1)*500)+"BrPo"+str(j)+".png", dpi=300)
                    plto.close()
                    if j==1:
                        breakpointsList1[0][i] = pw_results["estimates"]["breakpoint1"]["estimate"]
                        breakpointsList1[1][i] = lastT
    
                        alpha_hat1 = pw_fit.best_muggeo.best_fit.raw_params[1]
                        beta_hats1 = pw_fit.best_muggeo.best_fit.raw_params[2:3]
                        alphasList1[0][i] = alpha_hat1                                #[i] was needed write checl!
                        for k in range(1):
                            alphasList1[k+1][i] = alphasList1[k][i] + beta_hats1[k]
                    if j==2:
                        breakpointsList2[0][i] = pw_results["estimates"]["breakpoint1"]["estimate"]
                        breakpointsList2[1][i] = pw_results["estimates"]["breakpoint2"]["estimate"]
                        breakpointsList2[2][i] = lastT
    
                        alpha_hat2 = pw_fit.best_muggeo.best_fit.raw_params[1]
                        beta_hats2 = pw_fit.best_muggeo.best_fit.raw_params[2:4]
                        alphasList2[0][i] = alpha_hat2                               #[i] was needed write checl!
                        for k in range(2):
                            alphasList2[k+1][i] = alphasList2[k][i] + beta_hats2[k]
                    if j==3:
                        breakpointsList3[0][i] = pw_results["estimates"]["breakpoint1"]["estimate"]
                        breakpointsList3[1][i] = pw_results["estimates"]["breakpoint2"]["estimate"]
                        breakpointsList3[2][i] = pw_results["estimates"]["breakpoint3"]["estimate"]
                        breakpointsList3[3][i] = lastT
    
                        alpha_hat3 = pw_fit.best_muggeo.best_fit.raw_params[1]
                        beta_hats3 = pw_fit.best_muggeo.best_fit.raw_params[2:5]
                        alphasList3[0][i] = alpha_hat3                                #[i] was needed write checl!
                        for k in range(3):
                            alphasList3[k+1][i] = alphasList3[k][i] + beta_hats3[k]
            except Exception as err:
                print("______Error is caught _____"+str(i)+"__"+str(j)+"___")
                print(err)
                print("_____")
                BICsList[j][i] = np.inf 
                if j==1:                                    
                    breakpointsList1[0][i] = np.inf
                    breakpointsList1[1][i] = np.inf
                    for k in range(2):
                        alphasList1[k][i] = np.inf                       
                if j==2:                                    
                    breakpointsList2[0][i] = np.inf
                    breakpointsList2[1][i] = np.inf
                    breakpointsList2[2][i] = np.inf
                    for k in range(3):
                        alphasList2[k][i] = np.inf
                if j==3:                                    
                    breakpointsList3[0][i] = np.inf
                    breakpointsList3[1][i] = np.inf
                    breakpointsList3[2][i] = np.inf
                    breakpointsList3[3][i] = np.inf                
                    for k in range(4):
                        alphasList3[k][i] = np.inf            
    
    
                
    
    BICsList[-1][:] = np.argmin(BICsList, axis=0)             # Checks which Bic is smallest for model selection
    
    
    
    #saving to Tsv
    dataBIC = pd.DataFrame(BICsList)     
    
    
    dataAlphas0 = pd.DataFrame(alphasList0)
    dataAlphas1 = pd.DataFrame(alphasList1)
    dataBreakpoints1 = pd.DataFrame(breakpointsList1)
    
    dataAlphas2 = pd.DataFrame(alphasList2)
    dataBreakpoints2 = pd.DataFrame(breakpointsList2)
    
    dataAlphas3 = pd.DataFrame(alphasList3)
    dataBreakpoints3 = pd.DataFrame(breakpointsList3)
    
    
    
    
    #Save in here
    dataBIC.to_csv(path1+ '/BICs.tsv', sep="\t")
    dataAlphas0.to_csv(path1+'/Alphas0.tsv', sep="\t")   
    dataAlphas1.to_csv(path1+'/Alphas1.tsv', sep="\t")               
    dataBreakpoints1.to_csv(path1+'/Breakpoints1.tsv', sep="\t")
    dataAlphas2.to_csv(path1+'/Alphas2.tsv', sep="\t")               
    dataBreakpoints2.to_csv(path1+'/Breakpoints2.tsv', sep="\t")
    dataAlphas3.to_csv(path1+'/Alphas3.tsv', sep="\t")               
    dataBreakpoints3.to_csv(path1+'/Breakpoints3.tsv', sep="\t")
    
    #Save in Grand directory
    dataBIC.to_csv(path2+ '/BICs.tsv', sep="\t")
    dataAlphas0.to_csv(path2+'/Alphas0.tsv', sep="\t")  
    dataAlphas1.to_csv(path2+'/Alphas1.tsv', sep="\t")               
    dataBreakpoints1.to_csv(path2+'/Breakpoints1.tsv', sep="\t")
    dataAlphas2.to_csv(path2+'/Alphas2.tsv', sep="\t")               
    dataBreakpoints2.to_csv(path2+'/Breakpoints2.tsv', sep="\t")
    dataAlphas3.to_csv(path2+'/Alphas3.tsv', sep="\t")               
    dataBreakpoints3.to_csv(path2+'/Breakpoints3.tsv', sep="\t")
    


velocitydata = pd.read_table("Velocities_25or30.txt", names=["filename", "v", "beta", "k", "bool"])
velocitydataSize = len(velocitydata)

# Number of threads (adjust based on available resources)
num_threads = 25

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_file, velocitydata["filename"][i], velocitydata["k"][i], velocitydata["beta"][i]) for i in range(1,velocitydataSize)]
    # Wait for all tasks to complete
    for future in futures:
        future.result()

print("All files processed.")