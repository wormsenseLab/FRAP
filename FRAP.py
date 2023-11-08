import imageio
import os
import fnmatch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas
from scipy.signal import peak_widths
from scipy.ndimage import gaussian_filter
import datetime

from scipy import stats

from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
window = Tk()
window.wm_attributes('-topmost', 1) #this displays the tk window on top of other windows
window.withdraw()   # this supress the tk window


#%%

#specify columns of the pandas dataframe and excel sheets
cols_Data =     ['Date', 'Strain', 'Label', 'Allele', 'ImageID', 'Time(s)', 'Raw bl_roi intensity', 'Raw nonbl_roi intensity', 'FRAP intensity']
cols_Analysis = ['Date', 'Strain', 'Label', 'Allele', 'ImageID', 'Max length traced (px)', 'Pre-bleach intensity', 'Post-bleach time', 'ROI position', 'ROI width(px)', 'Recovery at 60s', 'Recovery at 300s']
cols_Rejects =  ['ImageID', 'Reason']

strain_key=pandas.DataFrame({('GN753', 'mNG::MEC-4', 'WT', 0),
                             ('GN935', 'mNG::MEC-4', 'mec-9(u437)', 0),
                             ('GN922', 'mNG::MEC-4', 'mec-1(e1738)', 0),
                             ('TV2411', 'myr::GFP', 'WT', 0),
                             ('NK2335', 'LAM-2::mNG', 'WT', 0),
                             ('NK2443', 'NID-1::mNG', 'WT', 0),
                             ('NK2413', 'SDN-1::mNG', 'WT', 0)}, columns=['Strain','Label','Allele', 'n'])
strain_key=strain_key.set_index('Strain')

umperpx=0.283

#%%
fpath = askdirectory(title='Select folder where straightened images are stored',parent=window)+'/' # shows dialog box and return the path
dfpath = askdirectory(title='Select destination folder for saving analysis files',parent=window)+'/' # shows dialog box and return the path

toa = str(datetime.datetime.today()).split()
today = toa[0]
now = toa[1]
timestamp = today.replace('-','')+'-'+now.replace(':','')[:6]
os.mkdir(dfpath+timestamp+'_individual_traces')

imgfiles = fnmatch.filter(os.listdir(fpath), '*.tif')


#%%
#initialize Pandas DataFrames
df_Data = pandas.DataFrame()
df_Analysis = pandas.DataFrame()
df_Rejects = pandas.DataFrame()


for x in imgfiles:                                #create loop for number of images in folder
    imseq = imageio.get_reader(fpath+x)           #import image and store it in a list of lists
    
    #extract info from filename
    date=x.split('_')[0]
    strain = x.split('_')[1]
    label = strain_key.loc[strain,'Label']
    allele = strain_key.loc[strain,'Allele']
    
    if label=='myr::GFP':
        time = np.append(np.arange(-10,0,1), np.arange(0,60,1))       # pre-bleach 10 frames @ 1 sec interval, post-bleach  @ 1 sec interval
    elif x.split('_')[2].split(' ')[0]=='longFRAP':
        time = np.append(np.arange(-10,0,1), np.arange(0,60*15,15))     # pre-bleach 10 frames @ 1 sec interval, post-bleach  @ 15 sec interval
    else:
        time = np.append(np.arange(-10,0,1), np.arange(0,60*5,5))       # pre-bleach 10 frames @ 1 sec interval, post-bleach  @ 5 sec interval
    
    
    for frame in imseq:
        columns=np.shape(frame)[1]
        break
    
    
    #convert imseq data structure to 3D-numpy array [[time-points],[rows],[columns]]
    imarray=np.zeros((70,21,columns))
    c=0
    for frame in imseq:
        imarray[c]=imarray[c]+frame
        c=c+1
    
    #detecting the bleached region by subtracting the mean of the first 3 post-bleach frames from the mean of the last three pre-bleach frames
    dif = gaussian_filter(np.mean(imarray[7:10,:,:], axis=(0,1))-np.mean(imarray[10:13,:,:], axis=(0,1)), sigma=3)
    bl_pos=np.argmax(dif)
    
    #compute bleached area width and edges
    roiw = peak_widths(dif, [bl_pos], rel_height=0.5)
    bleach_start = int(roiw[2][0])
    bleach_end = int(roiw[3][0])

    #reject if bleach region is too close to image edges
    if bleach_start<7 or bleach_end>columns-7:
        reason = 'Bleach region too close to edges'
        frame = pandas.DataFrame([[x, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(frame)
        continue
        
    #reject if detected bleach region is less than 5 pixels
    if roiw[0][0]<5:
        reason = 'Too narrow bleach region'
        frame = pandas.DataFrame([[x, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(frame)
        continue
    
    #designate cell, bleach and non-bleach regions    
    roi_cell = imarray[:,8:13,:]
    roi_bl = roi_cell[:,:,bleach_start:bleach_end]
    roi_nonbl = np.concatenate((roi_cell[:,:,:bleach_start-5], roi_cell[:,:,bleach_end+5:]), axis=2)

    # reject if there are saturated pixels in the pre-bleach region
    n_satpx = np.count_nonzero(roi_bl>=65520)
    n_totalpx = np.prod(np.shape(roi_bl))
    if n_satpx>0.00*n_totalpx:
        reason = 'Saturated pixels in bleach region'
        df = pandas.DataFrame([[x, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(df)
        continue    
        
    # reject if there are more than permissible (>5% of all pixels) saturated pixels in the whole cell
    n_satpx = np.count_nonzero(roi_cell>=65520)
    n_totalpx = np.prod(np.shape(roi_cell))
    if n_satpx>0.05*n_totalpx:
        reason = 'Too many saturated pixels in whole cell'
        df = pandas.DataFrame([[x, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(df)
        continue     

    #calculate average intensities of the bleached area over time
    bl = np.mean(roi_bl, axis=(1,2))        #Raw intensity over time of bleached area.
    bl_pre = np.mean(bl[:10])               #Average intensity of the first 10 frames. Used for quality control only. Not used for further calculation.
    bl_norm = (bl-bl[10])/(bl_pre-bl[10])   #Normalized intensity over time and scaled to set intensity at first post-bleach time point to zero. Used for plotting only. Not used for further calculation.

    #reject if drop in intensity in the roi is less than half of pre-bleach intensity
    if (bl_pre-bl[10])<0.5*bl_pre:
        reason = 'Not significant bleaching'
        frame = pandas.DataFrame([[x, reason]], columns=cols_Rejects)
        df_Rejects = df_Rejects.append(frame)
        continue


    nonbl = np.mean(roi_nonbl, axis=(1,2))  #Raw intensity over time of non-bleached area
    nonbl_pre = np.mean(nonbl[:10])         #Average intensity of the first 10 frames. Not used for further calculation.
    nonbl_norm = nonbl/nonbl_pre            #Normalized intensity over time. Used for plotting only. Not used for further calculation.

    # blcorr = bl/nonbl                       #Intensity corrected for overall photobleaching over time
    blcorr = bl*nonbl[0]/nonbl                       #Intensity corrected for overall photobleaching over time
    blcorr_pre = np.mean(blcorr[:10])       #Average corrected intensity of the first 10 frames. 
    blcorr_norm = (blcorr-blcorr[10])/(blcorr_pre-blcorr[10])       #Normalized corrected intensity over time and scaled to set corrected intensity at first post-bleach time point to zero.
    

    count = strain_key.loc[strain,'n'] + 1
    strain_key.at[strain,'n']=count
    

    # #create figure for the kymograph
    plt.figure(1, figsize=(0.02*columns, 15))
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['svg.fonttype'] = 'none'

    grid = plt.GridSpec(5, 2, wspace=0.5, hspace=0.5)
    
    plt.subplot(grid[0, 0:])
    plt.title(x)
    kymo = np.mean(roi_cell, axis=1)
    plt.imshow(kymo)
    
    xaxis=np.arange(0, columns)*umperpx
    
    plt.subplot(grid[1, 0:])
    plt.plot(xaxis, np.mean(kymo[:10,:], axis=0))
    plt.xlim(0,columns*umperpx)
    plt.ylim(0,65520)
    plt.vlines(np.array([bleach_start, bleach_end])*umperpx,0,65520, color='r',alpha=0.3)
    plt.xlabel('Distance from cell body (um)')
    plt.ylabel('Pre-bleach int (AU)')      
    
    plt.subplot(grid[2, 0:])
    plt.plot(xaxis, dif)
    plt.plot(bl_pos*umperpx, dif[bl_pos], 'go')
    plt.xlim(0,columns*umperpx)
    plt.vlines(np.array([bleach_start, bleach_end])*umperpx,np.min(dif), np.max(dif), color='r',alpha=0.3)
    plt.xlabel('Distance from cell body (um)')
    plt.ylabel('Int diff (AU)')      
    
    plt.subplot(grid[3:, 0])
    plt.plot(time, bl, 'b-')
    plt.plot(time, nonbl, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (AU)')
   
    plt.subplot(grid[3:, 1])
    plt.plot(time, bl_norm, 'b-')
    plt.plot(time, nonbl_norm, 'g-')
    plt.plot(time, blcorr_norm, 'k-')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized corrected intensity')

    sns.set_style('white')
    sns.set_style('ticks', {'xtick.direction': 'in', 'ytick.direction': 'in'})
    sns.despine(offset=2, trim=False)

    plt.savefig(dfpath+timestamp+'_individual_traces/'+x+'_analysis.png')
    plt.show()
    plt.close()

    if label=='myr::GFP':
        r60s=np.mean(blcorr_norm[[60,65,69]])
        r300s=np.nan
    else:
        r60s=np.mean(blcorr_norm[20:23])
        r300s=np.mean(blcorr_norm[67:])


    # add image data to pandas dataframe
    all_data1 = pandas.DataFrame({'Date':[date]*70, 'Strain':[strain]*70, 'Label':label, 'Allele':[allele]*70, 'ImageID':[x]*70, 'Time(s)':time, 'Raw bl_roi intensity':bl, 'Raw nonbl_roi intensity':nonbl, 'FRAP intensity':blcorr_norm}, columns=cols_Data)
    df_Data=df_Data.append(all_data1)
    frame = pandas.DataFrame([[date, strain, label, allele, x, columns, blcorr_pre, time[-1], bl_pos, roiw[0][0], r60s, r300s]], columns=cols_Analysis)
    df_Analysis = df_Analysis.append(frame)


#%%
#save data to excel file
wb = pandas.ExcelWriter(dfpath+timestamp+'_Analysis.xlsx', engine='xlsxwriter')
df_Analysis.to_excel(wb, sheet_name='Analysis')
df_Rejects.to_excel(wb, sheet_name='Rejects')
strain_key.to_excel(wb, sheet_name='Strain summary')
wb.save()

df_Data.to_pickle(dfpath+timestamp+'_Data.pkl')
df_Analysis.to_pickle(dfpath+timestamp+'_Analysis.pkl')

