
# coding: utf-8

# In[ ]:

# import necessary modules
#get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
import math
from matplotlib import rc
import matplotlib.patches as patches

from scipy.interpolate import interp1d
from matplotlib.ticker import FixedLocator
from math import floor
from mpl_toolkits.axes_grid1 import make_axes_locatable

# In[ ]:

# esthetic definitions for the plots

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
# matplotlib.rc('font', **font)
matplotlib.mathtext.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]

#
# l_TT_low,Dl_TT_low,err_TT_low= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-TT-loL-full_R2.02.txt",unpack=True,usecols=(0,1,2))
# l_TE_low,Dl_TE_low,err_TE_low= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-TE-loL-full_R2.02.txt",unpack=True,usecols=(0,1,2))
# l_TT_high,Dl_TT_high,err_TT_high= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-TT-hiL-binned_R2.02.txt",unpack=True,usecols=(0,3,4))
# l_TE_high,Dl_TE_high,err_TE_high= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-TE-hiL-binned_R2.02.txt",unpack=True,usecols=(0,3,4))
# l_EE_low,Dl_EE_low,err_EE_low= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-EE-loL-full_R2.02.txt",unpack=True,usecols=(0,1,2))
# l_EE_high,Dl_EE_high,err_EE_high= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/COM_PowerSpect_CMB-EE-hiL-binned_R2.02.txt",unpack=True,usecols=(0,3,4))
# lmin_phiphi,lmax_phiphi,cl_phiphi,err_phiphi= np.loadtxt("/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/error_Planck/agressive_lensing.csv",unpack=True)

# In[ ]:
#set general configuration
common_settings = {'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'l_max_scalars':2600,
                   'n_s':0.9663,
                   'ln10^{10}A_s':3.045,
                   'tau_reio':0.055,
                   'omega_b':0.02242,
                   '100*theta_s':1.042059,
                   'P_k_max_1/Mpc':1.0,
                   'z_max_pk' : 5.0
                   }

# #%% compute reference LCDM
#
M = Class()
#remember that base lcdm model features one massive neutrino of 0.06 eV
print("~~~~~computing reference LCDM~~~~~")
M.set(common_settings)
#we consider three massive degenerate neutrinos
M.set({
'omega_b': 2.235636e-02,
'omega_cdm': 1.199407e-01 , #Omega_cdm 0.2626
'100*theta_s':1.041632e+00,
'ln10^{10}A_s':3.025991e+00,
'n_s':9.619666e-01,
'tau_reio':5.465042e-02,
'N_ncdm':1,
'N_ur':2.0328,
'deg_ncdm':1,
'm_ncdm':0.06,
'DMDE_interaction':0,
'w0_fld':-0.999,
'wa_fld':0,
'Omega_Lambda':0,
'use_ppf':'no',
'input_verbose':10,
'background_verbose':10,
'output_verbose':10,
'fluid_equation_of_state':'CLP',
'gauge':'newtonian'})
M.compute()



h = M.h() # get reduced Hubble for conversions to 1/Mpc

#derived = M.get_current_derived_parameters(['sigma8','Omega_m'])
#print("Omega_m for LCDM is %f" %derived['Omega_m'])


M_DMDE = Class()

#%% compute best-fit DCDM
print("~~~~~time =%.f s; computing our code~~~~~"%(timeafterref-start_time))
M_DMDE.set(common_settings)
#we consider three massive degenerate neutrinos

M_DMDE.set({
'omega_b': 2.239330e-02,
'omega_cdm': 1.195019e-01, #Omega_cdm 0.2626
'N_ncdm':1,
'100*theta_s':1.041884e+00,
'ln10^{10}A_s':3.043829e+00,
'n_s':9.657928e-01,
'tau_reio':5.465042e-02,
# 'background_ncdm_distribution': 0,
'N_ur':2.0328,
'deg_ncdm':1,
'm_ncdm':0.06,
'DMDE_interaction':9.037873e-04,
'DMDE_interaction_pow':2,
'w0_fld':-0.999,
'wa_fld':0,
'Omega_Lambda':0,
'use_ppf':'no',
'input_verbose':10,
'background_verbose':10,
'output_verbose':10,
'fluid_equation_of_state':'CLP',
'gauge':'newtonian'
})

M_DMDE.compute()
# coding: utf-8
# In[ ]:

# esthetic definitions for the plots

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
# matplotlib.rc('font', **font)
matplotlib.mathtext.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]
###planck 2018: problem, only the large l are binned. lowell unbinned?
#10^7 [l(l+1)]^2 C_l^phiphi / 2pi
Experiment,l_min,l_center,l_max,power,sigma_power= np.loadtxt("CMB_errors/Clphiphi_planck.txt",unpack=True)
lTT,DlTT_mean,DlTT_error_minus,DlTT_error_plus,DlTT_bestfit= np.loadtxt("CMB_errors/Planck2018_errorTT.txt",unpack=True)
lEE,DlEE_mean,DlEE_error_minus,DlEE_error_plus,DlEE_bestfit= np.loadtxt("CMB_errors/Planck2018_errorEE.txt",unpack=True)
lTE,DlTE_mean,DlTE_error_minus,DlTE_error_plus,DlTE_bestfit= np.loadtxt("CMB_errors/Planck2018_errorTE.txt",unpack=True)
lTT_ACT,DlTT_ACT_mean,DlTT_ACT_error= np.loadtxt("CMB_errors/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_TT_cmbonly.txt",unpack=True)
lEE_ACT,DlEE_ACT_mean,DlEE_ACT_error= np.loadtxt("CMB_errors/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_EE_cmbonly.txt",unpack=True)
lTE_ACT,DlTE_ACT_mean,DlTE_ACT_error= np.loadtxt("CMB_errors/cmbonly_spectra_dr4.01/act_dr4.01_D_ell_TE_cmbonly.txt",unpack=True)
lTE_SPT,DlTE_SPT_mean,DlTE_SPT_error,lEE_SPT,DlEE_SPT_mean,DlEE_SPT_error= np.loadtxt("CMB_errors/SPT3G.txt",unpack=True)
lmin,lmax,lTT_SPTPol,DlTT_SPTPol_mean,DlTT_SPTPol_error,lEE_SPTPol,DlEE_SPTPol_mean,DlEE_SPTPol_error,lTE_SPTPol,DlTE_SPTPol_mean,DlTE_SPTPol_error= np.loadtxt("CMB_errors/SPTPol.txt",unpack=True)
# lEDE2,TTEDE2,EEEDE2,BBEDE2,TEEDE2,dd,dT,dE = np.loadtxt("output/bestfit_ACT02_cl_lensed.dat",unpack=True)
#Columns are (1) ell_min, (2) ell_max
#(3) TT ell_center, (4) TT in D_ell [uK^2], (5) TT error in D_ell [uK^2]
#(6) EE ell_center, (7) EE in D_ell [uK^2], (8) EE error in D_ell [uK^2]
#(9) TE ell_center, (10) TE in D_ell [uK^2], (11) TE error in D_ell [uK^2]
#ref_M0p999= np.loadtxt("output/dcdmdr_G10000_M0p999_100000bins_cl_lensed.dat")

# In[ ]:

##############
#
#############################################
#
# Fixed settings
#
#
kvec = np.logspace(-4,np.log10(3),1)
legarray = []
twopi = 2.*math.pi
#
# Create figures
#
# fig_Pk, ax_Pk = plt.subplots()
# fig_TT, ax_TT = plt.subplots()
# fig_EE, ax_EE = plt.subplots()
# fig_PP, ax_PP = plt.subplots()

# fig_TT, (ax_TT, ax_EE, ax_PP, ax_Pk) = plt.subplots(4,1, sharex=False,gridspec_kw=dict(height_ratios=[1,1,1,1]),figsize=(35,40))
# fig_TT, (ax_TT, ax_Pk) = plt.subplots(2,1, sharex=False,gridspec_kw=dict(height_ratios=[1,1]),figsize=(15,20))

####OLD#####
# fig_TT, (ax_TT, ax_EE) = plt.subplots(2,1, sharex=True,gridspec_kw=dict(height_ratios=[1,1]),figsize=(15,20))
# fig_TT.subplots_adjust(hspace=0)
# ax_TT.tick_params(axis='x',labelsize=30,length=20,which='both',width=2,direction='inout')
# ax_TT.tick_params(axis='y',labelsize=30,length=20,which='both',width=2,direction='inout')
# ax_EE.tick_params(axis='x',labelsize=30,length=20,which='both',width=2,direction='inout')
# ax_EE.tick_params(axis='y',labelsize=30,length=20,which='both',width=2,direction='inout')
# # ax_Pk.tick_params(axis='x',labelsize=30,length=20,which='both',width=2,direction='inout')
# # ax_Pk.tick_params(axis='y',labelsize=30,length=20,which='both',width=2,direction='inout')
# ax_TT.tick_params(axis='x',labelsize=30,length=10,which='minor',direction='inout')
# ax_TT.tick_params(axis='y',labelsize=30,length=10,which='minor',direction='inout')
# ax_EE.tick_params(axis='x',labelsize=30,length=10,which='minor',direction='inout')
# ax_EE.tick_params(axis='y',labelsize=30,length=10,which='minor',direction='inout')
# ######
# ax_Pk.tick_params(axis='x',labelsize=30,length=10,which='minor',direction='inout')
# ax_Pk.tick_params(axis='y',labelsize=30,length=10,which='minor',direction='inout')

#ax_EE_log = plt.subplot(313)
#ax_TE_log = plt.subplot(312)
#ax_TT_log = plt.subplot(311)
fig_TT, (ax_TT_log, ax_TE_log, ax_EE_log, ax_pp_log) = plt.subplots(4,1,sharex=True)
# ax_TT_log.set_title('axEDE residuals w/r to $\Lambda$CDM bestfit to Planck',fontsize=13)

plt.subplots_adjust(hspace=0)
plt.setp(ax_TT_log.get_xticklabels(), fontsize=15)
plt.setp(ax_TE_log.get_xticklabels(), fontsize=15)
plt.setp(ax_EE_log.get_xticklabels(), fontsize=15)
plt.setp(ax_pp_log.get_xticklabels(), fontsize=15)
plt.setp(ax_TT_log.get_yticklabels(), fontsize=15)
plt.setp(ax_TE_log.get_yticklabels(), fontsize=15)
plt.setp(ax_EE_log.get_yticklabels(), fontsize=15)
plt.setp(ax_pp_log.get_yticklabels(), fontsize=15)
# plt.subplots_adjust(hspace=0)
# divider = make_axes_locatable(ax_TT_log)
# ax_TT_lin = divider.append_axes("right", size=5, pad=0)
#
# divider = make_axes_locatable(ax_TE_log)
# ax_TE_lin = divider.append_axes("right", size=5, pad=0)
#
# divider = make_axes_locatable(ax_EE_log)
# ax_EE_lin = divider.append_axes("right", size=5, pad=0)


# plt.setp(ax_TT_lin.get_yticklabels(), fontsize=15)
# plt.setp(ax_TE_lin.get_yticklabels(), fontsize=15)
# plt.setp(ax_EE_lin.get_yticklabels(), fontsize=15)
# plt.setp(ax_TT_lin.get_xticklabels(), fontsize=15)
# plt.setp(ax_TE_lin.get_xticklabels(), fontsize=15)
# plt.setp(ax_EE_lin.get_xticklabels(), fontsize=15)

conversion = pow(2.7255*1.e6,2)
# one_k = all_k['scalar'][0]     # this contains only the scalar perturbations for the requested k values


clM = M.lensed_cl(2600)
ll_LCDM = clM['ell'][2:]
clTT_LCDM = clM['tt'][2:]
clEE_LCDM = clM['ee'][2:]
clTE_LCDM = clM['te'][2:]
clPP_LCDM = clM['pp'][2:]
# ax_TT.semilogx(ll_LCDM,(ll_LCDM)*(ll_LCDM+1)/(2*np.pi)*(clTT_LCDM),'k',lw=5)
# ax_EE.semilogx(ll_LCDM,(ll_LCDM)*(ll_LCDM+1)/(2*np.pi)*(clEE_LCDM),'k',lw=5)
# ax_PP.semilogx(ll_LCDM,(ll_LCDM)*(ll_LCDM+1)/(2*np.pi)*(clPP_LCDM),'k',lw=5)


#     i=i+1
fTT = interp1d(ll_LCDM,clTT_LCDM)
fEE = interp1d(ll_LCDM,clEE_LCDM)
fTE = interp1d(ll_LCDM,clTE_LCDM)
fPP = interp1d(ll_LCDM,clPP_LCDM)
T_cmb = 2.7225 #we change units for Planck

fTT_ref = interp1d(ll_LCDM,clTT_LCDM*(ll_LCDM)*(ll_LCDM+1)/2/np.pi*(T_cmb*1.e6)**2)
fEE_ref = interp1d(ll_LCDM,clEE_LCDM*(ll_LCDM)*(ll_LCDM+1)/2/np.pi*(T_cmb*1.e6)**2)
fTE_ref = interp1d(ll_LCDM,clTE_LCDM*(ll_LCDM)*(ll_LCDM+1)/2/np.pi*(T_cmb*1.e6)**2)
fPP_ref = interp1d(ll_LCDM,10**7*clPP_LCDM*((ll_LCDM)*(ll_LCDM+1))**2/2/np.pi)

# 10^7 [l(l+1)]^2 C_l^phiphi / 2pi


# #OLD LCDM
# M.set({
# '100*theta_s':1.04077,
# 'omega_b':0.02225,
# 'omega_cdm':0.1198,
# 'ln10^{10}A_s':3.094,
# 'n_s':0.9645,
# 'tau_reio':0.079,
# 'do_shooting':'yes'
# })
#

#
# get Cls
#
# clM = M.raw_cl(2500
clM = M_DMDE.lensed_cl(2600)
ll = clM['ell'][2:]
clTT = clM['tt'][2:]
clTE = clM['te'][2:]
clEE = clM['ee'][2:]
clPP = clM['pp'][2:]

# lmax = max(l)
# lmin = min(l)
# l=[]
# l.append(lmin)
# l.append(2*lmax)

# ax_TT_lin.fill_between(l,-2,2,alpha=var_alpha, facecolor=var_color,
#                  linewidth=0)
# ax_TE_lin.fill_between(l,-2,2,alpha=var_alpha, facecolor=var_color,
#                  linewidth=0)
# ax_EE_lin.fill_between(l,-2,2,alpha=var_alpha, facecolor=var_color,
#                  linewidth=0)
def binned_Cl(Cl,central_l,width):
    # central_l = l_ini+width/2
    l_ini= central_l-width/2
    weight_total = 0
    result = 0
    Clb = 0
    # for i in range(0,int(width)):
    #     weight_total += (l_ini+i)*(l_ini+i+1)
    #     result += np.sqrt(2/(2*(l_ini+float(i))+1))*(l_ini+float(i))*(l_ini+float(i)+1)
    # print l_ini,l_ini+width,result, weight_total
    # return result/weight_total/np.sqrt(2)/np.pi
    for i in range(0,int(width)):
        weight_total += (l_ini+i)*(l_ini+i+1)
        Clb += (l_ini+float(i))*(l_ini+float(i)+1)*Cl(l_ini+i)
    # print(l_ini,l_ini+width,Clb)
    return Clb/weight_total

fTT_DMDE = interp1d(ll,clTT)
fEE_DMDE = interp1d(ll,clEE)
fTE_DMDE = interp1d(ll,clTE)
fpp_DMDE = interp1d(ll,clPP)

sigma_CV = np.sqrt(fTT(ll)*fEE(ll)+fTE(ll)**2)
# print(((clTE)-(fTE_DMDE(ll)))/sigma_CV,np.sqrt(1/(2*ll+1)),np.sqrt(fTT(ll)*fEE(ll)+fTE(ll)**2))
# ax_TT_log.plot(ll,(fTT_DMDE(ll)-fTTACT(ll))/fTTACT(ll),'r',lw=1.5)
# ax_TT_log.plot(lTT,(binned_Cl(fTT_DMDE,lTT,30)-binned_Cl(fTT,lTT,30))/binned_Cl(fTT,lTT,30),'k--',lw=1.5)
ax_TT_log.plot(ll,(fTT_DMDE(ll)-fTT(ll))/fTT(ll),'r',lw=1.5,alpha=1,label='DMDE interaction')
ax_TT_log.plot(ll,(fTT(ll)-fTT(ll))/fTT(ll),'k',lw=1.5,alpha=0.5,label=r'$\Lambda$CDM')
# ax_TE_log.plot(ll,((fTE_DMDE(ll))-(fTEACT(ll)))/sigma_CV,'r',lw=1.5,label='WMAP+ACT bestfit')
ax_TE_log.plot(ll,((fTE_DMDE(ll))-(fTE(ll)))/sigma_CV,'r',lw=1.5,alpha=1)
ax_TE_log.plot(ll,((fTE(ll))-(fTE(ll)))/sigma_CV,'k',lw=1.5,alpha=0.5)
# ax_EE_log.plot(ll,(fEE_DMDE(ll)-fEE(ll))/fEE(ll),'r',lw=1.5,label='Planck $< 1000$ +ACT bestfit')
# ax_EE_log.plot(ll,(fEE_DMDE(ll)-fEEACT(ll))/fEEACT(ll),'r',lw=1.5)
# ax_EE_log.plot(lEE,(binned_Cl(fEE_DMDE,lEE,30)-binned_Cl(fEE,lEE,30))/binned_Cl(fEE,lEE,30),'k--',lw=1.5,label='Planck+ACT bestfit (binned)')
ax_EE_log.plot(ll,(fEE_DMDE(ll)-fEE(ll))/fEE(ll),'r',lw=1.5,alpha=1)
ax_EE_log.plot(ll,(fEE(ll)-fEE(ll))/fEE(ll),'k',lw=1,alpha=0.5)
ax_pp_log.plot(ll,(fpp_DMDE(ll)-fPP(ll))/fPP(ll),'r',lw=1.5,alpha=1)
ax_pp_log.plot(ll,(fPP(ll)-fPP(ll))/fPP(ll),'k',lw=1,alpha=0.5)





# sigma_CV = np.sqrt(fTT(ll)*fEE(ll)+fTE(ll)**2)
# print(((clTE)-(fTE(ll)))/sigma_CV,np.sqrt(1/(2*ll+1)),np.sqrt(fTT(ll)*fEE(ll)+fTE(ll)**2))
# ax_TT_log.plot(ll,(fTT_DMDE(ll)-fTT(ll))/fTT(ll),'k--',lw=1.5)
# ax_TE_log.plot(ll,((fTE_DMDE(ll))-(fTE(ll)))/sigma_CV,'k--',lw=1.5,label='Planck TT bestfit')
# ax_EE_log.plot(ll,(fEE_DMDE(ll)-fEE(ll))/fEE(ll),'k--',lw=1.5)

#ax_TT_log.errorbar(lTT_ACT, DlTT_ACT_mean/fTT_DMDE(lTT_ACT)-1, yerr=[DlTT_ACT_error/DlTT_ACT_mean,DlTT_ACT_error/DlTT_ACT_mean], fmt='.',color='blue',alpha=0.6)
#ax_EE_log.errorbar(lEE_ACT, DlEE_ACT_mean/fEE_DMDE(lEE_ACT)-1, yerr=[DlEE_ACT_error/DlEE_ACT_mean,DlEE_ACT_errorDlEE_ACT_mean],fmt='.',color='blue',alpha=0.6,label='ACT')
#ax_TE_log.errorbar(lTE_ACT, DlTE_ACT_mean/fTE_DMDE(lTE_ACT)-1, yerr=[DlTE_ACT_error/DlTE_ACT_mean,DlTE_ACT_error/DlTE_ACT_mean], fmt='.',color='blue',alpha=0.6)


M.struct_cleanup()

#####
###NEW SCALES###
ax_TT_log.set_yscale('linear')
ax_TT_log.set_xlim((2,2600))
ax_TT_log.set_ylim((-0.2,0.2))
ax_EE_log.set_xscale('linear')
ax_EE_log.set_xlim((2,2600))
ax_EE_log.set_ylim((-0.2,0.2))
ax_TE_log.set_xscale('linear')
ax_TE_log.set_xlim((2,2600))
ax_TE_log.set_ylim((-0.2,0.2))
ax_pp_log.set_xlim((2,2600))
ax_pp_log.set_ylim((-0.2,0.2))
ax_TE_log.set_xscale('log')

# ax_TT_log.set_xticks([2., 10.])
# ax_EE_log.set_xticks([2., 10.])
# ax_TE_log.set_xticks([2., 10.])
# ax_TT_log.set_yticks([-0.8,-0.4,0.,0.4,0.8])
# ax_EE_log.set_yticks([-7.5,-5,-2.5,0.0,2.5,5,7.5])
# ax_TE_log.set_yticks([-5,-2.5,0.0,2.5,5])

# ax_TT_log.spines['right'].set_visible(False)
ax_EE_log.yaxis.set_ticks_position('left')
ax_TE_log.yaxis.set_ticks_position('left')
# ax_TT_log.spines['right'].set_visible(False)
ax_EE_log.yaxis.set_ticks_position('left')
ax_TE_log.yaxis.set_ticks_position('left')
ax_TT_log.tick_params('both', length=10, width=1, which='major')
ax_TT_log.tick_params('both', length=5, width=1, which='minor')
ax_EE_log.tick_params('both', length=10, width=1, which='major')
ax_EE_log.tick_params('both', length=5, width=1, which='minor')
ax_TE_log.tick_params('both', length=10, width=1, which='major')
ax_TE_log.tick_params('both', length=5, width=1, which='minor')

# ###ERROR BARS W/R LCDM###
# T_cmb = 2.7225
# factor1 = l_TT_high*(l_TT_high+1)/2./np.pi;
# conversion1 = 1/(factor1*(T_cmb*1.e6)**2)
# factor2 = l_TT_low*(l_TT_low+1)/2./np.pi;
# conversion2 = 1/(factor2*(T_cmb*1.e6)**2)
# factor3 = l_EE_high*(l_EE_high+1)/2./np.pi;
# conversion3 = 1/(factor3*(T_cmb*1.e6)**2)
# factor4 = l_EE_low*(l_EE_low+1)/2./np.pi;
# conversion4 = 1/(factor4*(T_cmb*1.e6)**2)
# factor5 = l_TE_high*(l_TE_high+1)/2./np.pi;
# conversion5 = 1/(factor3*(T_cmb*1.e6)**2)
# factor6 = l_TE_low*(l_TE_low+1)/2./np.pi;
# conversion6 = 1/(factor4*(T_cmb*1.e6)**2)

ax_TT_log.errorbar(lTT, DlTT_mean/fTT_ref(lTT)-1, yerr=[DlTT_error_minus/fTT_ref(lTT),DlTT_error_plus/fTT_ref(lTT)], fmt='.',color='gray')
ax_EE_log.errorbar(lEE, DlEE_mean/fEE_ref(lEE)-1, yerr=[DlEE_error_minus/fEE_ref(lEE),DlEE_error_plus/fEE_ref(lEE)], fmt='.',color='gray',label='{\it Planc} data')
ax_TE_log.errorbar(lTE, DlTE_mean/fTE_ref(lTE)-1, yerr=[DlTE_error_minus/fTE_ref(lTE),DlTE_error_plus/fTE_ref(lTE)], fmt='.',color='gray')
ax_pp_log.errorbar(l_center, power/fPP_ref(l_center)-1, yerr=[sigma_power/fPP_ref(l_center),sigma_power/fPP_ref(l_center)], fmt='.',color='gray')
print(l_center,power,fPP_ref(l_center))
print(lTE,DlTE_error_minus/fTE_ref(lTE))
# ax_EE_log.errorbar(lEE_SPT, DlEE_SPT_mean/fEE_ref(lEE_SPT)-1, yerr=[DlEE_SPT_error/DlEE_SPT_mean,DlEE_SPT_error/DlEE_SPT_mean], fmt='.',color='green',alpha=0.6,label='SPT3G')
# ax_TE_log.errorbar(lTE_SPT, DlTE_SPT_mean/fTE_ref(lTE_SPT)-1, yerr=[DlTE_SPT_error/DlTE_SPT_mean,DlTE_SPT_error/DlTE_SPT_mean], fmt='.',color='green',alpha=0.6)
# ax_EE_log.errorbar(lEE_SPTPol, DlEE_SPTPol_mean/fEE_ref(lEE_SPTPol)-1, yerr=[DlEE_SPTPol_error/DlEE_SPTPol_mean,DlEE_SPTPol_error/DlEE_SPTPol_mean], fmt='.',color='orange',alpha=0.6,label='SPTPol')
# ax_TE_log.errorbar(lTE_SPTPol, DlTE_SPTPol_mean/fTE_ref(lTE_SPTPol)-1, yerr=[DlTE_SPTPol_error/DlTE_SPTPol_mean,DlTE_SPTPol_error/DlTE_SPTPol_mean], fmt='.',color='orange',alpha=0.6)
# ax_TT_log.errorbar(lTT_SPTPol, DlTT_SPTPol_mean/fTT_ref(lTT_SPTPol)-1, yerr=[DlTT_SPTPol_error/DlTT_SPTPol_mean,DlTT_SPTPol_error/DlTT_SPTPol_mean], fmt='.',color='orange',alpha=0.6)




# ax_EE.fill_between(l_EE_low,-err_EE_low/Dl_EE_low,err_EE_low/Dl_EE_low,facecolor='grey',alpha=0.2)
#
#
# ax_TT.axis([2,2500,-0.06,0.06])
# ax_TT.set_xlabel(r'$\ell$',fontsize=35)
# ax_TT_lin.text(1050,0.02,r'$\frac{\Delta C_\ell^\mathrm{TT}}{C_\ell^\mathrm{TT}(\Lambda{\rm CDM})}$',fontsize=20)
# ax_TT_log.set_ylabel(r'$\Delta C_\ell^\mathrm{TT}/C_\ell^\mathrm{TT}(\Lambda{\rm CDM})$',fontsize=17)
ax_TT_log.set_ylabel(r'$\frac{\Delta C_\ell^\mathrm{TT}}{C_\ell^\mathrm{TT}}$',fontsize=19)

# ax_Pk.legend(frameon=False,prop={'size':30},loc='upper left',borderaxespad=0.)
ax_TT_log.set_xlabel(r'$\ell$',fontsize=20,labelpad=0)
# ax_EE_lin.legend(frameon=False,prop={'size':12},loc='upper right',borderaxespad=0.)

# ax_EE.axis([2,2500,-0.06,0.06])
ax_EE_log.set_xlabel(r'$\ell$',fontsize=20,labelpad=0)
# ax_EE_lin.text(200,-0.1,r'$\frac{\Delta C_\ell^\mathrm{EE}}{C_\ell^\mathrm{EE}(\Lambda{\rm CDM})}$',fontsize=20)
# ax_EE_log.set_ylabel(r'$\Delta C_\ell^\mathrm{EE}/C_\ell^\mathrm{EE}(\Lambda{\rm CDM})$',fontsize=19)
ax_EE_log.set_ylabel(r'$\frac{\Delta C_\ell^\mathrm{EE}}{C_\ell^\mathrm{EE}}$',fontsize=19)

ax_TT_log.legend(frameon=False,prop={'size':12},loc='lower left',borderaxespad=0.)
ax_TE_log.legend(frameon=False,prop={'size':12},loc='upper left',borderaxespad=0.)
# ax_EE_log.legend(frameon=False,prop={'size':12},loc='upper left',borderaxespad=0.)

# ax_TE.axis([2,2500,-0.06,0.06])
ax_TE_log.set_xlabel(r'$\ell$',fontsize=20,labelpad=0)
# ax_TE_lin.text(200,-0.1,r'$\frac{\Delta C_\ell^\mathrm{TE}}{C_\ell^\mathrm{TE}(\Lambda{\rm CDM})}$',fontsize=20)
# ax_TE_log.set_ylabel(r'$\Delta C_\ell^\mathrm{TE}/C_\ell^\mathrm{TE}(\Lambda{\rm CDM})$',fontsize=19)
ax_TE_log.set_ylabel(r'$\frac{\Delta C_\ell^\mathrm{TE}}{\sqrt{C_\ell^\mathrm{EE}C_\ell^\mathrm{TT}+(C_\ell^\mathrm{TE})^2}}$',fontsize=19)
ax_pp_log.set_xlabel(r'$\ell$',fontsize=20,labelpad=0)
ax_pp_log.set_ylabel(r'$\frac{\Delta C_\ell^\mathrm{\phi\phi}}{C_\ell^\mathrm{\phi\phi}}$',fontsize=19)

# ax_TE_lin.text(200,-0.1,r'$\frac{\Delta C_\ell^\mathrm{TE}}{C_\ell^\mathrm{TE}(\Lambda{\rm CDM})}$',fontsize=20)
# ax_TE_log.set_ylabel(r'$\Delta C_\ell^\mathrm{TE}/C_\ell^\mathrm{TE}(\Lambda{\rm CDM})$',fontsize=19)
# ax_pp_log.set_ylabel(r'$\frac{\Delta C_\ell^{\phi\phi}}{\sqrt{C_\ell^\mathrm{EE}C_\ell^\mathrm{TT}+(C_\ell^\mathrm{TE})^2}}$',fontsize=19)


plt.savefig('DMDE-vs_CMBData.pdf', bbox_inches='tight')


# In[ ]:




# In[ ]:

# In[ ]:
