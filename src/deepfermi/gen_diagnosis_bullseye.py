import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaHelperfunctions import segmentLabel
from visualisation.ahaBullseye.ahaHelperfunctions import PlottingSettings

class PatientDiagnosis():

    def __init__(self, pat_id, plottingSettings, ischemic_seg=[]):
        
        # General configurations
        self.pat_id = pat_id
        self.ischemic_seg = [x-1 for x in ischemic_seg] if ischemic_seg else ischemic_seg        
        self.bulls_eye = self.calc_bulls_eye(plottingSettings)
        
    def calc_bulls_eye(self,plottingSettings):
        """
        Bullseye representation for the left ventricle.

        Notes
        -----
        This function create the 17 segment model for the left ventricle according
        to the American Heart Association (AHA) [1]_

        References
        ----------
        .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
            S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
            and M. S. Verani, "Standardized myocardial segmentation and
            nomenclature for tomographic imaging of the heart",
            Circulation, vol. 105, no. 4, pp. 539-542, 2002.
        """
        
        # Initializing Polar plot
        fig = plt.figure(figsize=(4.8 , 4.8))
        ax = plt.subplot2grid((1,1),(0,0),projection='polar')
        ax.axes.cla()

        # General initialization
        data_avg = np.zeros(17)
        data_std = np.zeros(17)        
        cmap = plottingSettings.cmap
        vmin = plottingSettings.vmin
        vmax = plottingSettings.vmax
        show_segmentNumbers = plottingSettings.show_segmentNumbers
        show_std = plottingSettings.show_std
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Setting ischemic segments to one
        data_avg[self.ischemic_seg] = 1

        linewidth = 2
        data_avg = np.array(data_avg).ravel()

        theta = np.linspace(0, 2 * np.pi, 768)
        r = np.linspace(0.2, 1, 4)

        # Create the bound for the segment 17
        for i in range(r.shape[0]):
            ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

        # Create the bounds for the segments 1-12
        for i in range(6):
            theta_i = np.deg2rad(i * 60)
            ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

        # Create the bounds for the segments 13-16
        for i in range(4):
            theta_i = np.deg2rad(i * 90 - 45)
            ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)

        rot = [0.0, 60.0, -60.0, 0.0, 60.0, -60.0]
        size_font = 15

        # Fill the segments 1-6
        r0 = r[2:4]
        r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
        for i in range(6):
            # First segment start at 60 degrees
            theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
            theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
            z = np.ones((128, 2)) * data_avg[i]
            ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm,shading='gouraud')

            stringToAnnot = segmentLabel(offset=0,
                                        index = i,
                                        data_avg= data_avg,
                                        data_std = data_std,
                                        show_segmentNumber = show_segmentNumbers,
                                        show_std=show_std,
                                        newLine_afterNum=show_segmentNumbers and not (i == 0 or i == 3),
                                        newLine_beforeStd= False,
                                        show_val=False)

            ax.annotate(stringToAnnot,
                        xy=(theta[int(i * 128)] + np.deg2rad(90), 1.18 * (r[2])),  # theta, radius
                        ha='center',size = size_font,  va = 'center', rotation = rot[i]
                        )

        # Fill the segments 7-12
        r0 = r[1:3]
        r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
        for i in range(6):
            # First segment start at 60 degrees
            theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
            theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
            z = np.ones((128, 2)) * data_avg[i + 6]
            ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm,shading='gouraud')

            stringToAnnot = segmentLabel(offset=6,
                                        index=i,
                                        data_avg=data_avg,
                                        data_std=data_std,
                                        show_segmentNumber=show_segmentNumbers,
                                        show_std=show_std,
                                        newLine_afterNum=show_segmentNumbers and not (i == 0 or i == 3),
                                        newLine_beforeStd= False,
                                        show_val=False)

            ax.annotate(stringToAnnot,
                        xy=(theta[int(i * 128)] + np.deg2rad(90), 1.3 * (r[1])),  # theta, radius
                        ha='center',size = size_font,  va = 'center', rotation = rot[i]
                        )

        # Fill the segments 13-16
        r0 = r[0:2]
        r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
        for i in range(4):
            # First segment start at 45 degrees
            theta0 = theta[i * 192:i * 192 + 192] + np.deg2rad(45)
            theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
            z = np.ones((192, 2)) * data_avg[i + 12]
            ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm,shading='gouraud')

            stringToAnnot = segmentLabel(offset=12,
                                        index=i,
                                        data_avg=data_avg,
                                        data_std=data_std,
                                        show_segmentNumber=show_segmentNumbers,
                                        show_std=show_std,
                                        newLine_afterNum=show_segmentNumbers and i % 2 != 0,
                                        show_val=False)

            ax.annotate(stringToAnnot,
                        xy=(theta[int(i * 192 )]+ np.deg2rad(90) , 1.6*(r[0])),  # theta, radius
                        ha='center',size = size_font,  va = 'center'
                        )


        # Fill the segments 17
        if data_avg.size == 17:
            r0 = np.array([0, r[0]])
            r0 = np.repeat(r0[:, np.newaxis], theta.size, axis=1).T
            theta0 = np.repeat(theta[:, np.newaxis], 2, axis=1)
            z = np.ones((theta.size, 2)) * np.nan
            ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm,shading='gouraud')

        ax.set_ylim([0, 1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        fig.tight_layout(pad=0)

        return ax

# Patient diagnosis
dic = {}
dic['10'] = {}
dic['10']['ischemic_seg']=[13, 7]
dic['36'] = {}
dic['36']['ischemic_seg']=[15,9,10,3,4,5]
dic['37'] = {}
dic['37']['ischemic_seg']=[4,5,6]
dic['38'] = {}
dic['38']['ischemic_seg']=[]
dic['39'] = {}
dic['39']['ischemic_seg']=[]
dic['42'] = {}
dic['42']['ischemic_seg']=[]
dic['44'] = {}
dic['44']['ischemic_seg']=[15,10,4]
dic['45'] = {}
dic['45']['ischemic_seg']=[13,14,7,8,2]
dic['46'] = {}
dic['46']['ischemic_seg']=[]
dic['47'] = {}
dic['47']['ischemic_seg']=[]
dic['54'] = {}
dic['54']['ischemic_seg']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
dic['56'] = {}
dic['56']['ischemic_seg']=[]
dic['58'] = {}
dic['58']['ischemic_seg']=[]
dic['61'] = {}
dic['61']['ischemic_seg']=[13,14,7,8,9,1,2]
dic['62'] = {}
dic['62']['ischemic_seg']=[]
dic['63'] = {}
dic['63']['ischemic_seg']=[1,2,7,8,12,13,14,16]
dic['71'] = {}
dic['71']['ischemic_seg']=[]
dic['75'] = {}
dic['75']['ischemic_seg']=[]
dic['76'] = {}
dic['76']['ischemic_seg']=[]
dic['83'] = {}
dic['83']['ischemic_seg']=[7,8,9,10,11,12,13,14,15,16]
dic['86'] = {}
dic['86']['ischemic_seg']=[]
dic['101'] = {}
dic['101']['ischemic_seg']=[]
dic['102'] = {}
dic['102']['ischemic_seg']=[]
dic['107'] = {}
dic['107']['ischemic_seg']=[1,2,3,7,8,9]
dic['108'] = {}
dic['108']['ischemic_seg']=[1,2,3,4,7,8,9,10,13,14,15]#[1,3,4,5,6,7,9,10,11,12,15,16]#
dic['109'] = {}
dic['109']['ischemic_seg']=[]

# Settings for bulls eye plot
plottingSettings = PlottingSettings(cmap = plt.cm.viridis,
                                    vmin = 0,
                                    vmax = 1.0,
                                    show_segmentNumbers = True,
                                    show_std = False,
                                    closePlotAutomatically = True,
                                    show_debuggingImages = False,
                                    useEdgesToSetInnerPoints= False)

# Save path
perf_aha_path = Path('/data/brahma01/Datasets/perfusion_kcl/aha/')

for p in tqdm(dic.keys()):
    ischemic_seg = dic[p]['ischemic_seg']
    pdiag = PatientDiagnosis(p, plottingSettings, ischemic_seg=ischemic_seg)
    # Saving diagnosis
    pdata_path = Path.joinpath(perf_aha_path, p + '_STRESS_moco')
    diagnosis = np.zeros(17)
    diagnosis[pdiag.ischemic_seg] = 1
    np.save(Path.joinpath(pdata_path, 'diagnosis.npy'), diagnosis)
    pdiag.bulls_eye.figure.savefig(Path.joinpath(pdata_path, 'diagnosis.png'), dpi=300)



