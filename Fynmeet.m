clear all;
close all;
%% Global Variables
for i=1:1
global path
global RangeBin
global R
global f
global A
global lambda
global SS
global pol
global DataSet
global GrazingAngle
global elevation 
global azimuth
global theta_w
global swh
global wndspd
global data 
global x 
global p_mle_rayl
global p_mom_rayl
global p_mle_logn
global p_mom_logn
global p_mle_wbl
global p_mom_wbl
global p_watts_k
global p_mom_k
global p_r_k
global modChiSqr_rayl_mle
global modChiSqr_rayl_mom
global modChiSqr_logn_mle
global modChiSqr_logn_mom
global modChiSqr_wbl_mle
global modChiSqr_wbl_mom
global modChiSqr_k_watts
global modChiSqr_k_mom
global modChiSqr_k_r
end
%% Load Dataset
% loads a variables Cdata, Info, NumOfPRIs, NumOfRangeBins, PCI, PRI_s, Waveform, lamda  
% Choose between CFA17_001, CFA17_002, CFA17_003, CFC17_001 , CFC14_004 ,
% CFC16_001
DataSetName = '11_011_CS';
load(DataSetName);
DataSet = DataSetName(7:9);
%% Range Bin
RangeBin = 1;               % Specify range bin to process 
%% Path for Figures
path='C:\Users\micro\.QtWebEngineProcess\Desktop\EEE4022S\Figures';
%% Radar parameters
c = 299792458;                      % Wavelength of the radar
Bandwidth_Hz = 10e6;                         % Pulsewidth = 100us, pure sinusoid at 9GHz
RangeResolution_m = c/(2*Bandwidth_Hz);      % Range resolution in meters 
for setParameters = 1:1
    if DataSetName == 'CFA17_001'
        pol = 'VV';
        f = 6.9;
        GrazingAngle = (1.02+1.27)/2; 
        wndspd = 14.5; %kts
        wind_gust = 23.7; %kts
        avg_wind = 13.8; %kts
        swh = 2.31; %m
        azimuth = 2;
        ant_az = 180.7;
        elevation = -1.225; %degrees
        R = PCI.Range;
        theta_w = 253.4 -ant_az;
        lambda = c/(f*10^9);
    elseif DataSetName =='CFA17_002'
        pol = 'VV';
        f = 6.9;
        GrazingAngle = (0.472+0.525)/2;
        wndspd = 14.4; %kts
        wind_gust = 23.7; %kts
        avg_wind = 13.8; %kts
        swh = 2.31; %m
        azimuth = 2;
        ant_az = 180.7;
        elevation = -1.225; %degrees
        R = PCI.Range;
        theta_w = 253.4 -ant_az;
         lambda = c/(f*10^9);
    elseif DataSetName == 'CFA17_003'
        pol = 'VV';
        f = 6.9;
        GrazingAngle =(0.323+0.35)/2;
        inst_wind = 14.4; %kts
        wind_gust = 23.7; %kts
        avg_wind = 13.8; %kts
        swh = 2.3; %m
        azimuth = 2;
        ant_az = 180.7;
        elevation = -1.219; %degrees
        R = PCI.Range;
         lambda = c/(f*10^9);
    elseif DataSetName == 'CFC17_001'
        pol = 'VV';
        f = 9;
        GrazingAngle =(0.853+1.27)/2;
        wndspd = 15.5; %kts
        wind_gust = 29.2; %kts
        avg_wind = 11.4; %kts
        swh = 2.22; %m
        azimuth = 1.8;
        ant_az = 165.5;
        elevation = -1.06; %degrees
        R = PCI.Range;  
         lambda = c/(f*10^9);
    elseif DataSetName == '08_017_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.293 + 0.314 )/2;
        wndspd = 5.64; %kts
        wind_gust = 29.2; %kts
        avg_wind = 3.84; %kts
        swh = 2.44; %m
        azimuth = 1.8;
        ant_az = 240;
        elevation = -0.571; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
     elseif DataSetName == '08_018_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.293 + 0.314 )/2;
        wndspd = 5.64; %kts
        wind_gust = 29.2; %kts
        avg_wind = 3.84; %kts
        swh = 2.44; %m
        azimuth = 1.8;
        ant_az = 262.5;
        elevation = -0.571; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
      elseif DataSetName == '08_019_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.293 + 0.314 )/2;
        wndspd = 6.61; %kts
        wind_gust = 29.2; %kts
        avg_wind = 4.62; %kts
        swh = 2.21; %m
        azimuth = 1.8;
        ant_az = 285;
        elevation = -0.571; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
    elseif DataSetName == '08_020_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.293 + 0.314 )/2;
        wndspd = 6.61; %kts
        wind_gust = 29.2; %kts
        avg_wind = 4.62; %kts
        swh = 2.21; %m
        azimuth = 1.8;
        ant_az = 307.5;
        elevation = -0.569; %degrees
        R = PCI.RangeS;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
     elseif DataSetName == '08_021_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.293 + 0.314 )/2;
        wndspd = 6.61; %kts
        wind_gust = 29.2; %kts
        avg_wind = 4.62; %kts
        swh = 2.21; %m
        azimuth = 1.8;
        ant_az = 330;
        elevation = -0.5711; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
    elseif DataSetName == '11_007_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.0958 + 0.108)/2;
        wndspd = 15.6; %kts
        avg_wind = 8.29; %kts
        swh = 2.78; %m
        azimuth = 1.8;
        ant_az = 240;
        elevation = -0.4944; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
    elseif DataSetName == '11_008_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.0958 + 0.108)/2;
        wndspd = 15.6; %kts
        avg_wind = 8.29; %kts
        swh = 2.78; %m
        azimuth = 1.8;
        ant_az = 262.5;
        elevation = -0.4944; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);
     elseif DataSetName == '11_009_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.0958 + 0.108)/2;
        wndspd = 15.6; %kts
        avg_wind = 8.29; %kts
        swh = 2.78; %m
        azimuth = 1.8;
        ant_az = 285;
        elevation = -0.4944; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);    
     elseif DataSetName == '11_010_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.0958 + 0.108)/2;
        wndspd = 15.6; %kts
        avg_wind = 8.29; %kts
        swh = 2.48; %m
        azimuth = 1.8;
        ant_az = 307.5;
        elevation = -0.4944; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);    
    elseif DataSetName == '11_011_CS'
        pol = 'VV';
        f = 8.8;  
        GrazingAngle =(0.0958 + 0.108)/2;
        wndspd = 15.6; %kts
        avg_wind = 8.29; %kts
        swh = 2.48; %m
        azimuth = 1.8;
        ant_az = 330;
        elevation = -0.4944; %degrees
        R = PCI.Range;   
        A = 15*R*(azimuth*pi/180)*sec(GrazingAngle*pi/180);
        lambda = c/(f*10^9);    
    end 
end
%% Plot Cdata 
PRI = PRI_s; % TFC15_008
VRange_m = (1:1:size(Cdata,2)); 
VRangeBins = 1:1:size(Cdata,2);
TimeVector = (1:1:size(Cdata,1))*PRI;
%% Extract one range bin 

fs = 1/PRI;
ts = 1/fs;
NumRangeLines = size(Cdata,1);
StartRangeLine = round(NumRangeLines/2); 
StopRangeLine = round(NumRangeLines/4*3);
X = Cdata(StartRangeLine:StopRangeLine,RangeBin);

% DataOneBin = Cdata(:,RangeBin); %Extract the data from the specified bin only 
% DataMeanSubtracted = DataOneBin - mean(DataOneBin);  %Normalise about mean ***CHECK*** 
% data = abs(DataMeanSubtracted)'; % amplitude of the complex clutter
%% Reflectivity 

reflectivityCompare(Cdata);
%% Main - Amplitude Stats

%for RangeBin = 1:1:size(Cdata,2)

X = Cdata(StartRangeLine:StopRangeLine,RangeBin);

DataOneBin = Cdata(:,RangeBin); %Extract the data from the specified bin only 
DataMeanSubtracted = DataOneBin - mean(DataOneBin);  %Normalise about mean ***CHECK*** 
data = abs(DataMeanSubtracted)'; % amplitude of the complex clutter
% Plot histogram
sizeData = length(data);
data = sort(data,'ascend');
x = 0:0.0001:max(data)+0.5;
%Rayleigh
[p_mle_rayl,p_mom_rayl,sigma_mle_rayl,sigma_mom_rayl] = rayleighPDF(data);
[modChiSqr_rayl_mle, modChiSqr_rayl_mom] = modChiSqr_rayl(10,0.1,data,sigma_mle_rayl,sigma_mom_rayl); 
% Lognormal Distribution
[p_mle_logn,p_mom_logn,sigma_mle_logn,sigma_mom_logn,mu_mle_logn,mu_mom_logn] = lognPDF(data);
[modChiSqr_logn_mle, modChiSqr_logn_mom] = modChiSqr_logn(10,0.1,data,sigma_mle_logn,sigma_mom_logn,mu_mle_logn,mu_mom_logn); 
% Weibull Distribution 
[p_mle_wbl,p_mom_wbl,shape_mle_wbl,shape_mom_wbl,scale_mle_wbl,scale_mom_wbl] = wblPDF(data);
[modChiSqr_wbl_mle, modChiSqr_wbl_mom] = modChiSqr_wbl(10,0.1,data,shape_mle_wbl,shape_mom_wbl,scale_mle_wbl,scale_mom_wbl); 
% K = 10;
% PFA = 0.1;
% N=length(data);       %get length of sea clutter data
%     CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
%     stepSize =      PFA/K;
%     probability_intervals=  CDF_start:stepSize:1; 
%     %intervals_CDF=  0:0.1:1; 
%     x1 = 0:1e-4:data(N); %Get cdf of k distribution over the desired range
%     wbl_cdf_mle = 1 - exp(-(x1./scale_mle_wbl).^shape_mle_wbl);
%     wbl_cdf_mom = 1 - exp(-(x1./scale_mom_wbl).^shape_mom_wbl);
% for i=1:length(probability_intervals)
%         % find the index (b) of the value in the cdf that is associated
%         % with the probabilites stipulated by K --> then find the x value
%         % associated with this index
%         [a,b]=min(abs(wbl_cdf_mle-probability_intervals(i)));
%         xvals_invCDF_mle_wbl(i)=x1(b);
%         
%         [a,b]=min(abs(wbl_cdf_mom-probability_intervals(i)));
%         xvals_invCDF_mom_wbl(i)=x1(b);
% 
% end
%     % Since the x value associated with 100% of the cdf tends to infinity --> we couldn't however plot the cdf over infinite range 
%     %this value is the value at the end of the STATISTICAL MODEL of the cdf
%     %not the data!!
%     xvals_invCDF_mle_wbl(end)=Inf; 
%     xvals_invCDF_mom_wbl(end)=Inf;
% 
%     
%     %count how many x values are in these intervals for the actual data 
%     fi_mle_wbl= histc(data,xvals_invCDF_mle_wbl);
%     fi_mom_wbl= histc(data,xvals_invCDF_mom_wbl);
% 
%     fi_mle_wbl(end)=[];
%     fi_mom_wbl(end)=[];
%   
%     %Using Formula in (Chan):
%     global modChiSqr_wbl_mle
%     global modChiSqr_wbl_mom
%     modChiSqr_wbl_mle= sum((fi_mle_wbl-(PFA)*N/K).^2/((PFA)*N/K));
%     modChiSqr_wbl_mom= sum((fi_mom_wbl-(PFA)*N/K).^2/((PFA)*N/K));
   
%% K-Dsitribution
[p_watts_k,p_mom_k,p_r_k,shape_watts_k,shape_mom_k,shape_r_k,scale_watts_k,scale_mom_k,scale_r_k] = kPDF(data);

%[modChiSqr_k_watts, modChiSqr_k_mom, modChiSqr_k_r] = modChiSqr_k(10,0.1,data,shape_watts_k,shape_mom_k,shape_r_k,scale_watts_k,scale_mom_k,scale_r_k);
K = 10;
PFA = 0.1;
N=length(data);       %get length of sea clutter data
    CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize =      PFA/K;
    probability_intervals=  CDF_start:stepSize:1; 
    %intervals_CDF=  0:0.1:1; 
    x1 = 0:1e-4:data(N); %Get cdf of k distribution over the desired range
    k_cdf_watts= 1 - ((2/gamma(shape_watts_k))*((scale_watts_k.*x1./2).^shape_watts_k).*besselk(shape_watts_k,scale_watts_k.*x1)); %You dont need to change this
    k_cdf_mom= 1 - ((2/gamma(shape_mom_k))*((scale_mom_k.*x1./2).^shape_mom_k).*besselk(shape_mom_k,scale_mom_k.*x1)); %You dont need to change this
    k_cdf_r= 1 - ((2/gamma(shape_r_k))*((scale_r_k.*x1./2).^shape_r_k).*besselk(shape_r_k,scale_r_k.*x1)); %You dont need to change this
    
 for i=1:length(probability_intervals)
        % find the index (b) of the value in the cdf that is associated
        % with the probabilites stipulated by K --> then find the x value
        % associated with this index
        [a,b]=min(abs(k_cdf_watts-probability_intervals(i)));
        xvals_invCDF_watts(i)=x1(b);
        
        [a,b]=min(abs(k_cdf_mom-probability_intervals(i)));
        xvals_invCDF_mom(i)=x1(b);
        
        [a,b]=min(abs(k_cdf_r-probability_intervals(i)));
        xvals_invCDF_r(i)=x1(b);
end
    % Since the x value associated with 100% of the cdf tends to infinity --> we couldn't however plot the cdf over infinite range 
    %this value is the value at the end of the STATISTICAL MODEL of the cdf
    %not the data!!
    xvals_invCDF_watts(end)=Inf; 
    xvals_invCDF_mom(end)=Inf;
    xvals_invCDF_r(end)=Inf;
    
    %count how many x values are in these intervals for the actual data 
    fi_watts= histc(data,xvals_invCDF_watts);
    fi_mom= histc(data,xvals_invCDF_mom);
    fi_r= histc(data,xvals_invCDF_r);
    
    fi_watts(end)=[];
    fi_mom(end)=[];
    fi_r(end)=[];
    
    %Using Formula in (Chan 2006):
    modChiSqr_k_watts= round(sum((fi_watts-(PFA)*N/K).^2/((PFA)*N/K)));
    modChiSqr_k_mom= round(sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K)));
    modChiSqr_k_r= round(sum((fi_r-(PFA)*N/K).^2/((PFA)*N/K)));
    
%fprintf('\nModified Chi-Squared Test: \nRayleigh MLE: %i \nRayleigh MoM: %i\nLognormal MLE: %i \nLognormal MoM: %i \nWeibull MLE: %i \nWeibull MoM: %i \nK Watts: %i \nK MoM: %i \nK Rag: %i \n',modChiSqr_rayl_mle, modChiSqr_rayl_mom,modChiSqr_logn_mle, modChiSqr_logn_mom,modChiSqr_wbl_mle, modChiSqr_wbl_mom,modChiSqr_k_watts, modChiSqr_k_mom, modChiSqr_k_r);
i = RangeBin;
arr_modChiSqr_rayl_mle(i) = modChiSqr_rayl_mle;
arr_modChiSqr_rayl_mom(i)= modChiSqr_rayl_mom;
arr_modChiSqr_logn_mle(i)=modChiSqr_logn_mle;
arr_modChiSqr_logn_mom(i) = modChiSqr_logn_mom;
arr_modChiSqr_wbl_mle(i)= modChiSqr_wbl_mle;
arr_modChiSqr_wbl_mom(i)=modChiSqr_wbl_mom;
arr_modChiSqr_k_watts(i)=modChiSqr_k_watts;
arr_modChiSqr_k_mom(i)=modChiSqr_k_mom;
arr_modChiSqr_k_r(i)=modChiSqr_k_r;

%% Call to Plot 
% plotAll();      %plots all distributions 
% plotRay();      %plots Rayleigh MLE and MoM
% plotLogn();     %plots Lognormal MLE and MoM
% plotWbl();      %plots Weibull MLE and MoM
% plotK();        %plots K-Dsitribution MLE and MoM
% plotBest();       %plots the best fit of each dsitribution
% %close all;
% figure()
%end
%% Average Test over Dataset
av_modChiSqr_rayl_mle = mean(arr_modChiSqr_rayl_mle);
av_modChiSqr_rayl_mom=mean(arr_modChiSqr_rayl_mom);
av_modChiSqr_logn_mle=mean(arr_modChiSqr_logn_mle);
av_modChiSqr_logn_mom=mean(arr_modChiSqr_logn_mom);
av_modChiSqr_wbl_mle=mean(arr_modChiSqr_wbl_mle);
av_modChiSqr_wbl_mom=mean(arr_modChiSqr_wbl_mom);
av_modChiSqr_k_watts=mean(arr_modChiSqr_k_watts);
av_modChiSqr_k_mom=mean(arr_modChiSqr_k_mom);
av_modChiSqr_k_r=mean(arr_modChiSqr_k_r);

fprintf('\nAverage Modified Chi-Squared Test: \nRayleigh MLE: %i \nRayleigh MoM: %i\nLognormal MLE: %i \nLognormal MoM: %i \nWeibull MLE: %i \nWeibull MoM: %i \nK Watts: %i \nK MoM: %i \nK Rag: %i \n',av_modChiSqr_rayl_mle, av_modChiSqr_rayl_mom,av_modChiSqr_logn_mle, av_modChiSqr_logn_mom,av_modChiSqr_wbl_mle, av_modChiSqr_wbl_mom,av_modChiSqr_k_watts, av_modChiSqr_k_mom, av_modChiSqr_k_r);
%% Call to Plot 
% plotAll();      %plots all distributions 
% plotRay();      %plots Rayleigh MLE and MoM
% plotLogn();     %plots Lognormal MLE and MoM
% plotWbl();      %plots Weibull MLE and MoM
% plotK();        %plots K-Dsitribution MLE and MoM
% plotBest;       %plots the best fit of each dsitribution
% close all;
% figure()

%% Plotting + Saving Figures
function plotAll()
% Plot histogram
global path
global RangeBin
global GrazingAngle
global DataSet
global data 
global x 
global p_mle_rayl
global p_mom_rayl
global p_mle_logn
global p_mom_logn
global p_mle_wbl
global p_mom_wbl
global p_watts_k
global p_mom_k
global p_r_k

figure(1);
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;
%Rayleigh
plot(x,p_mle_rayl,'Linewidth',1,'LineStyle','--');
hold on;
plot(x,p_mom_rayl,'Linewidth',1,'LineStyle','--');
hold on;
%Lognormal
plot(x,p_mle_logn,'Linewidth',1,'LineStyle',':');
hold on;
plot(x,p_mom_logn,'Linewidth',1,'LineStyle',':');
hold on;
%weibull 
plot(x,p_mle_wbl,'Linewidth',1,'LineStyle','-.');
hold on;
plot(x,p_mom_wbl,'Linewidth',1,'LineStyle','-.');
hold on;
% K 
plot(x,p_watts_k,'Linewidth',1);
hold on;
plot(x,p_mom_k,'Linewidth',1);
hold on;
plot(x,p_r_k,'Linewidth',1);
%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data','Rayleigh MLE','Rayleigh MoM','Lognormal MLE','Lognormal MoM','Weibull MLE','Weibull MoM','K-Distribution Watts','K-Distribution MoM','K-Distribution Raghavan','Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %f.3',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_All',DataSet,RangeBin);
saveas(figure(1),fullfile(path,[baseFileName '.png']));
end 
function plotRay()
global path
global RangeBin 
global GrazingAngle
global DataSet
global data 
global x 
global p_mle_rayl
global p_mom_rayl

figure(2)
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;
%Rayleigh
plot(x,p_mle_rayl,'Linewidth',1,'LineStyle','--');
hold on;
plot(x,p_mom_rayl,'Linewidth',1,'LineStyle','--');

%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data','Rayleigh MLE','Rayleigh MoM','Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %.3f',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_Rayl',DataSet,RangeBin);
saveas(figure(2),fullfile(path,[baseFileName '.png']));
end 
function plotLogn()
% Plot histogram
global path
global RangeBin 
global GrazingAngle
global DataSet
global data 
global x 
global p_mle_logn
global p_mom_logn
figure(3)
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;

%Lognormal
plot(x,p_mle_logn,'Linewidth',1,'LineStyle',':');
hold on;
plot(x,p_mom_logn,'Linewidth',1,'LineStyle',':');

%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data','Lognormal MLE','Lognormal MoM','Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %.3f',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_LogN',DataSet,RangeBin);
saveas(figure(3),fullfile(path,[baseFileName '.png']));
end 
function plotWbl()
% Plot histogram
global path
global RangeBin 
global GrazingAngle
global DataSet
global data 
global x 
global p_mle_wbl
global p_mom_wbl

figure(4)
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;
%weibull 
plot(x,p_mle_wbl,'Linewidth',1,'LineStyle','-.');
hold on;
plot(x,p_mom_wbl,'Linewidth',1,'LineStyle','-.');

%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data','Weibull MLE','Weibull MoM','Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %.3f',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_Wbl',DataSet,RangeBin);
saveas(figure(4),fullfile(path,[baseFileName '.png']));
end 
function plotK()
% Plot histogram
global path
global RangeBin 
global GrazingAngle
global DataSet
global data 
global x 
global p_watts_k
global p_mom_k
global p_r_k

figure(5)
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;

% K 
plot(x,p_watts_k,'Linewidth',1);
hold on;
plot(x,p_mom_k,'Linewidth',1);
hold on;
plot(x,p_r_k,'Linewidth',1);
%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data','K-Distribution Watts','K-Distribution MoM','K-Distribution Raghavan','Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %.3f',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_K',DataSet,RangeBin);
saveas(figure(5),fullfile(path,[baseFileName '.png']));
end 
function plotBest()
% Plot histogram
global path
global RangeBin 
global GrazingAngle
global DataSet
global data 
global x 
global p_mle_rayl
global p_mom_rayl
global p_mle_logn
global p_mom_logn
global p_mle_wbl
global p_mom_wbl
global p_watts_k
global p_mom_k
global p_r_k
global modChiSqr_rayl_mle
global modChiSqr_rayl_mom
global modChiSqr_logn_mle
global modChiSqr_logn_mom
global modChiSqr_wbl_mle
global modChiSqr_wbl_mom
global modChiSqr_k_watts
global modChiSqr_k_mom
global modChiSqr_k_r

figure(6)
%Data
[No,edges] = histcounts(data,100, 'Normalization','pdf');
edges = edges(2:end) - (edges(2)-edges(1))/2;
scatter(edges, No,'x','k','LineWidth',0.8);
set(gca,'YScale','log')
hold on;

if modChiSqr_rayl_mle < modChiSqr_rayl_mom
    l1 = 'Rayleigh MLE';
    plot(x,p_mle_rayl,'Linewidth',1,'LineStyle','--');
    hold on;
else
    l1 = 'Rayleigh MoM';
    plot(x,p_mom_rayl,'Linewidth',1,'LineStyle','--');
    hold on;
end 

if modChiSqr_logn_mle < modChiSqr_logn_mom
    l2 = 'Lognormal MLE';
    plot(x,p_mle_logn,'Linewidth',1,'LineStyle',':');
    hold on;
else
    l2 = 'Lognormal MoM';
    plot(x,p_mom_logn,'Linewidth',1,'LineStyle',':');
    hold on;
end 

if (modChiSqr_wbl_mle < modChiSqr_wbl_mom)
    l3 = 'Weibull MLE';
    plot(x,p_mle_wbl,'Linewidth',1,'LineStyle','-.');
    hold on;
else
    l3 = 'Weibull MoM';
    plot(x,p_mom_wbl,'Linewidth',1,'LineStyle','-.');
    hold on;
end 

if (modChiSqr_k_watts < modChiSqr_k_mom) && (modChiSqr_k_watts < modChiSqr_k_r)
    l4 = 'K-Distribution Watts';
    plot(x,p_watts_k,'Linewidth',1);
elseif (modChiSqr_k_mom < modChiSqr_k_watts) && (modChiSqr_k_mom < modChiSqr_k_r)
    l4 = 'K-Distribution MoM';
    plot(x,p_mom_k,'Linewidth',1);
elseif (modChiSqr_k_r < modChiSqr_k_watts) && (modChiSqr_k_r < modChiSqr_k_mom)
    l4 = 'K-Distribution Raghavan';
    plot(x,p_r_k,'Linewidth',1);
end 

%Plot
xlim([0 5])
ylim([0.000001 10])
xlabel('Normalised Amplitude');
ylabel('Probability');
legend('Data',l1,l2,l3,l4,'Location','southwest');
t1 = sprintf('Range Bin: %i  Grazing Angle: %.3f',RangeBin, GrazingAngle);
title(t1);
baseFileName = sprintf('%s_%i_Best',DataSet,RangeBin);
saveas(figure(6),fullfile(path,[baseFileName '.png']));
end 
%% Amplitude Distribution Fitting 
function [p_mle,p_mom,sigma_mle,sigma_mom] = rayleighPDF(data)
    N = numel(data);
    global x
    %MLE: Get MLE sigma estimate
    sigma_mle = raylfit(data);
    %MoM: Get MoM sigma estimate
    sigma_mom = mean(data)*sqrt(2)/sqrt(pi); %From https://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_rayleigh_theory.html
    %Get PDF Function
    p_mle =(x./sigma_mle^2).*exp(-(x.^2)/(2*sigma_mle^2));
    p_mom= (x./sigma_mom^2).*exp(-(x.^2)/(2*sigma_mom^2));
end
function [p_mle,p_mom,sigma_mle,sigma_mom,mu_mle,mu_mom] = lognPDF(data)
    N = numel(data);
    
    %MLE: Get MLE sigma estimate
    parameters= lognfit(data);
    mu_mle=     parameters(1);
    sigma_mle=  parameters(2);
    %MoM: Get MoM sigma estimate
    mu_mom=     -0.5*log(sum(data.^2))+2*log(sum(data))-(3/2)*log(N);
    sigma_mom=  sqrt(log(sum(data.^2))-2*log(sum(data))+log(N));
    %Get PDF Function
    global x
    p_mle = (1./(x.*sigma_mle*sqrt(2*pi))).*exp((-(log(x-mu_mle)).^2)/(2*sigma_mle^2));
    p_mom = (1./(x.*sigma_mom*sqrt(2*pi))).*exp((-(log(x-mu_mom)).^2)/(2*sigma_mom^2));
end
function [p_mle,p_mom,shape_mle,shape_mom,scale_mle,scale_mom] = wblPDF(data)
    N = numel(data);
    
    %MLE: Get MLE sigma estimate
    parameters= wblfit(data);
    scale_mle=  parameters(1);
    shape_mle=  parameters(2);
    %MoM: Get MoM sigma estimate 
    m1 = mean(data);            %first sample moment
    m2 = (1/N)*sum(data.^2);    %second sample moment
    
    fun = @(est_shape) (gamma(1+(2/est_shape))/(gamma(1+(1/est_shape))^2))-(m2/(m1^2));
    
    samp_var = (1/(N-1))*sum((data-mean(data)).^2);       %sample variance
    c0 = (mean(data)/sqrt(samp_var))^1.086;                         %define starting point
    
    shape_mom = fzero(fun,c0);
    scale_mom =mean(data)/gamma(1+(1/shape_mom));
    
    %Get PDF Function
    global x
    p_mle = (shape_mle/scale_mle).*(x./scale_mle).^(shape_mle-1).*exp(-(x./scale_mle).^shape_mle);
    p_mom = (shape_mom/scale_mom).*(x./scale_mom).^(shape_mom-1).*exp(-(x./scale_mom).^shape_mom);
end
function [p_watts,p_mom,p_r,shape_watts,shape_mom,shape_r,scale_watts,scale_mom,scale_r] = kPDF(data)
    N = numel(data);
    global RangeBin 
    %Watts's Method: Get shape and scale estimate using 2nd and 4th moments
    m2 = (1/N)*sum(data.^2); %second sample moment
    m4 = (1/N)*sum(data.^4); %fourth sample moment
    
    shape_watts = ((m4/(2*(m2)^2))-1)^(-1);
    scale_watts = 2*sqrt(shape_watts/m2);

    %MoM: Get shape and scale estimate using 1st and 2nd moments
    m1 = mean(data);        %first sample moment
    m2 = mean(data.^2);     %second sample moment
    if (RangeBin == 18) || (RangeBin == 19)|| (RangeBin == 21)|| (RangeBin == 48)
        shape_mom = shape_watts;
        scale_mom = scale_watts;
    else 
        fun = @(v)(4.*v.*gamma(v).^2)./(pi.*gamma(v+0.5).^2) -m2./(m1.^2);
        shape_mom= fzero(fun,shape_watts);
        scale_mom = gamma(shape_mom+0.5)*sqrt(pi)/(gamma(shape_mom)*m1);
    end 
    %Raghavan's Method
    shape_r = rag(data);
    scale_r = (2/mean(data))*(gamma(shape_r+0.5)*gamma(1.5))/(gamma(shape_r));
    
    %Get PDF Function
    global x
    p_watts= (2*scale_watts/gamma(shape_watts)).*((0.5*scale_watts.*x).^shape_watts).*besselk(shape_watts-1,scale_watts.*x);
    p_mom= (2*scale_mom/gamma(shape_mom)).*((0.5*scale_mom.*x).^shape_mom).*besselk(shape_mom-1,scale_mom.*x);
    p_r= (2*scale_r/gamma(shape_r)).*((0.5*scale_r.*x).*shape_r).*besselk(shape_r-1,scale_r.*x);
end
%% Modified Chi-Squared Tests
function [modChiSqr_mle, modChiSqr_mom]=modChiSqr_rayl(K,PFA,data,sigma_mle,sigma_mom)
    % K :       No. of even intervals to preak up region <PFA. 
    %PFA:       The amplitude region where Pfa <= (in Chan PFA = 0.1).
    % data:     The measured data the fit is being compared to
    
    N=length(data);         %get length of sea clutter data
    CDF_start=      1-PFA;  %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize=       PFA/K;
    intervals_CDF=  CDF_start:stepSize:1; 
    intervals_CDF=  0:0.1:1; 
    PFA = 1;
    interval_invCDF_mle = raylinv(intervals_CDF,sigma_mle);
    interval_invCDF_mom = raylinv(intervals_CDF,sigma_mom);
   
    %fi = how many elements in data betweeen the K intervals in interval_invCDF
    fi_mle= histc(data,interval_invCDF_mle); 
    fi_mom= histc(data,interval_invCDF_mom); 
    
    
    fi_mle(end)=[];
    fi_mom(end)=[];
     
    %Using Formula in (Chan):
    modChiSqr_mle= round(sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K)));
    modChiSqr_mom= round(sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K)));
    %NOrmal Chi Sqr

 
end 
function [modChiSqr_mle, modChiSqr_mom]=modChiSqr_logn(K,PFA,data,sigma_mle,sigma_mom,mu_mle,mu_mom)
    % K :       No. of even intervals to preak up region <PFA. 
    % PFA:      The amplitude region where Pfa <= (in Chan PFA = 0.1).
    % data:     The measured data the fit is being compared to
    
    N=length(data);       %get length of sea clutter data
    CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize =      PFA/K;
    intervals_CDF=  CDF_start:stepSize:1; 
    %intervals_CDF=  0:0.1:1; 
    PFA = 0.1;
    interval_invCDF_mle = logninv(intervals_CDF,mu_mle,sigma_mle);
    interval_invCDF_mom = logninv(intervals_CDF,mu_mom,sigma_mom);
    
    %fi = how many elements in data betweeen the K intervals in interval_invCDF
    fi_mle= histc(data,interval_invCDF_mle); 
    fi_mom= histc(data,interval_invCDF_mom); 
    fi_mle(end)=[];
    fi_mom(end)=[];
    
    %Using Formula in (Chan):
    modChiSqr_mle= round(sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K)));
    modChiSqr_mom= round(sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K)));
    
   

end 
function [modChiSqr_mle, modChiSqr_mom]=modChiSqr_wbl(K,PFA,data,shape_mle,shape_mom,scale_mle,scale_mom)
    % K :       No. of even intervals to preak up region <PFA. 
    % PFA:      The amplitude region where Pfa <= (in Chan PFA = 0.1).
    % data:     The measured data the fit is being compared to
    
    N=length(data);       %get length of sea clutter data
    CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize =      PFA/K;
    intervals_CDF=  CDF_start:stepSize:1; 
    %intervals_CDF=  0:0.1:1; 
    PFA = 0.1;
    interval_invCDF_mle = wblinv(intervals_CDF,scale_mle,shape_mle);
    interval_invCDF_mom = wblinv(intervals_CDF,scale_mom,shape_mom);
    
    %fi = how many elements in data betweeen the K intervals in interval_invCDF
    fi_mle= histc(data,interval_invCDF_mle); 
    fi_mom= histc(data,interval_invCDF_mom); 
    fi_mle(end)=[];
    fi_mom(end)=[];
    
    %Using Formula in (Chan):
    modChiSqr_mle= round(sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K)));
    modChiSqr_mom= round(sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K)));

end 
function [modChiSqr_watts, modChiSqr_mom, modChiSqr_r]=modChiSqr_k(K,PFA,data,shape_watts,shape_mom,shape_r,scale_watts,scale_mom,scale_r)
    % K :       No. of even intervals to preak up region <PFA. 
    % PFA:      The amplitude region where Pfa <= (in Chan PFA = 0.1).
    % data:     The measured data the fit is being compared to

    
    N=length(data);       %get length of sea clutter data
    CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize =      PFA/K;
    probability_intervals=  CDF_start:stepSize:1; 
    %intervals_CDF=  0:0.1:1; 
    x = 0:0.0001:data(N); %Get cdf of k distribution over the desired range
    k_cdf_watts= 1 - ((2/gamma(shape_watts)).*((scale_watts.*x./2).^shape_watts).*besselk(shape_watts,scale_watts.*x)); %You dont need to change this
    k_cdf_mom= 1 - ((2/gamma(shape_mom)).*((scale_mom.*x./2).^shape_mom).*besselk(shape_mom,scale_mom.*x)); %You dont need to change this
    k_cdf_r= 1 - ((2/gamma(shape_r)).*((scale_r.*x./2).^shape_r).*besselk(shape_r,scale_r.*x)); %You dont need to change this
    
    for i=1:length(probability_intervals)
        % find the index (b) of the value in the cdf that is associated
        % with the probabilites stipulated by K --> then find the x value
        % associated with this index
        [a,b]=min(abs(k_cdf_watts-probability_intervals(i)));
        xvals_invCDF_watts(i)=x(b);
        
        [a,b]=min(abs(k_cdf_mom-probability_intervals(i)));
        xvals_invCDF_mom(i)=x(b);
        
        [a,b]=min(abs(k_cdf_r-probability_intervals(i)));
        xvals_invCDF_r(i)=x(b);
    end
    % Since the x value associated with 100% of the cdf tends to infinity --> we couldn't however plot the cdf over infinite range 
    %this value is the value at the end of the STATISTICAL MODEL of the cdf
    %not the data!!
    xvals_invCDF_watts(end)=Inf; 
    xvals_invCDF_mom(end)=Inf;
    xvals_invCDF_r(end)=Inf;
    
    %count how many x values are in these intervals for the actual data 
    fi_watts= histc(data,xvals_invCDF_watts);
    fi_mom= histc(data,xvals_invCDF_mom);
    fi_r= histc(data,xvals_invCDF_r);
    
    fi_watts(end)=[];
    fi_mom(end)=[];
    fi_r(end)=[];

    %Using Formula in (Chan):
    modChiSqr_watts= sum((fi_watts-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_mom= sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_r= sum((fi_r-(PFA)*N/K).^2/((PFA)*N/K));
    
end 
%% Reflectivity 
function [mean_reflectivity] = getReflectivity(Cdata)
    global R
    global azimuth
    global GrazingAngle 
    global elevation
    global A
    fprintf('Measured Reflectivity \n');
    for RangeBin = 1:1:48
        DataOneBin = Cdata(:,RangeBin); %Extract the data from the specified bin only 
        DataMeanSubtracted = DataOneBin - mean(DataOneBin);  %Normalise about mean ***CHECK*** 
        data = abs(DataMeanSubtracted); % amplitude of the complex clutter
%         A = pi*R^2*tan((-elevation*(pi/180))/2)*tan((azimuth*(pi/180))/2)*csc(GrazingAngle*(pi/180));
       
        %A = 15*((pi/180)*azimuth*R);
        sigma = mean((abs(data)).^2);
        sigma_o_measured(RangeBin) = 10*log10(sigma/A);
        fprintf('Range bin: %i  Reflectivity: %f \n',RangeBin, sigma_o_measured(RangeBin));
        %R = R + 15;
    end
    fprintf('Average Reflectivity in Dataset: %f \n', round(mean(sigma_o_measured)));
    mean_reflectivity = round(mean(sigma_o_measured));
end
%GIT Model
function [mean_GIT] = GIT()
%Get the normalised clutter reflectivity from the GIT reflectivity model 
% gr : Grazing Angle in degrees to be converted to radians
% wndvel : wind velocity kts to be converted to kts 
% pol : radar polarization
% theta_w : look direction relative to wnd direction in degrees to be converted
% to radians
%swh : significant wave height in m 
% lambda : radar frequency in GHz
    global pol 
    global lambda
    global GrazingAngle
    global theta_w
    global swh 
    global SS
%     global pol 
    gr = GrazingAngle;
    
    %convert from degrees to radians
    gr = deg2rad(gr);
    theta_w = deg2rad(gr);

    %get average wave heigh hav from swh
    hav = 1.6*swh;
    U = 3.189*((SS)^(0.8));
    %Get adjustment factors Ga, Fu, Gw 
    a  = (14.4*lambda + 5.5)*((gr*hav)/(lambda));
    q = 1.1/((lambda + 0.015)^(0.4));

    Ga = (a^4)/(1+a^4);
    Gu = exp(0.2*cos(theta_w)*(1-2.8*gr)*(lambda + 0.015)^(-0.4));
    Gw = ((1.94*U)/(1+U/15.4))^q;

    HH = 10*log10(3.9*10^(-6)*lambda*(gr^0.4)*Ga*Gu*Gw);
    if pol == 'HH' 
        mean_GIT = HH;
    elseif pol == 'VV'
        mean_GIT = HH -1.05*log(hav+0.015) + 1.09*log(lambda) + 1.27*log(gr+0.0001) + 9.7;
    end 
end
% Hybrid Model 
function [mean_HYB] = HYB()
    global theta_w
    global f 
    global swh
    global GrazingAngle 
    global pol 
    global SS
    global lambda
    gr = GrazingAngle;

    %get reference reflectivity for SS 5, gr = 0.1, Pol = VV, theta_w = 0;
    sigma_ref = 24.4*log10(f) -65.2; % f in Ghz
    %get RMS wave height sigma_h
    sigma_h = 0.031*(SS^2); %this is the most common approximation used
    sigma_h = swh/2.83;
    %define reference angle in degrees
    phi_ref = 0.1;
    %define transitional angle in degrees
    phi_t = asin(0.0632*lambda/sigma_h)*(180/pi);

    if phi_t >= phi_ref
        if gr < phi_ref
            Kg = 0;
        elseif (phi_ref<=gr)&& (gr<=phi_t)
            Kg = 20*log10(gr/phi_ref);
        elseif (phi_t <= gr) && (gr<=30)
            Kg = 20*log10(phi_t/phi_ref) + 10*log10(gr/phi_t);
        end
    elseif phi_t < phi_ref
        if gr <= phi_ref
            Kg = 0;
        elseif gr>phi_ref
            Kg = 10*log10(gr/phi_ref);
        end
    end 

    Ks = 5*(SS-5);
    hav = 0.08*SS^2;
    if pol == 'HH'
        Kp = 1.1*log(hav + 0.015) - 1.1*log(lambda) -1.3*log(gr/57.3 + 0.0001) - 9.7;
    elseif pol == 'VV'
        Kp = 0;
    end

    Kd = (2+1.7*log10(0.1/lambda))*(cos(theta_w) - 1); 
    mean_HYB = sigma_ref + Kg+ Ks + Kp + Kd;
end
%TSC Model 
function [mean_TSC] = TSC()
    global lambda
    global theta_w 
    global GrazingAngle
    global f
    global pol 
    global SS
    gr = deg2rad(GrazingAngle);
    theta = deg2rad(theta_w);
    %get surface height standard deviation in m 
    sigma_z = 0.03505*SS^(1.95);
    %get sigma_a 
    sigma_a = 4.5416*gr*(3.2808*sigma_z+0.25)/lambda;
    Ga = sigma_a^1.5/(1+sigma_a^1.5);
    %get windspeed in m/s
    U = 3.189*(SS^0.8);
    %get constant terms
    Q = gr^0.6;

    A1 = (1+(lambda/0.00914)^3)^0.1;
    A2 = (1+(lambda/0.00305)^3)^0.1;
    A3 = (1+(lambda/0.00914)^3)^(Q/3);
    A4 = 1+0.35*Q;
    A = 2.63*A1/(A2*A3*A4);

    Gw =((1.9438*U+4)/15)^(A);
    Gu = exp(0.3*cos(theta)*exp(-gr/0.17)/(10.7636*lambda^2 + 0.005)^0.2);

    HH = 10*log10(1.7*10^(-5)*(gr^(0.5))*Gu*Gw*Ga/((3.2802*lambda + 0.05)^1.8));
    if pol == 'HH' 
        mean_TSC = 10*log10(1.7*10^(-5)*gr^(0.5)*Gu*Gw*Ga/(3.2802*lambda + 0.05)^1.8);
    elseif pol == 'VV' 
        mean_TSC = HH  -1.05*log(8.225*sigma_z+0.05) +1.09*log(lambda) + 1.27*log(sin(gr) + 0.0001)+10.945;
    end 
end
function [SS]  = Douglas()
% Get the Douglas sea state given the significant wave height h1/3
global swh
global SS
if (swh<0.333)
    SS = 1;
elseif (swh>=0.333) && (swh<0.9144)
    SS = 2;
elseif (swh>=0.9144) && (swh<1.5239)
    SS = 3;
elseif (swh>=1.5239) && (swh<2.4385)
    SS = 4;
elseif (swh>=2.4385) && (swh<3.657)
    SS = 5;
elseif (swh>=3.657) && (swh<6.0957)
    SS = 6;
elseif (swh>=6.0957) && (swh<12.1914)
    SS = 7;
end
end
function [] = reflectivityCompare(Cdata)
    global SS 
    global GrazingAngle
    gr = GrazingAngle;
    mean_reflectivity = getReflectivity(Cdata);
    SS  = Douglas();
    mean_GIT = GIT();
    mean_HYB = HYB();
    mean_TSC = TSC();
    fprintf('Reflecitvity over the dataset:\n Mean Measured Reflectivity: %f \nGIT Model: %f\nHYB Model: %f\nTSC Model: %f\n',mean_reflectivity,mean_GIT,mean_HYB,mean_TSC);
     [m,index] = min(abs([mean_GIT, mean_HYB, mean_TSC] - mean_reflectivity));
     if index == 1
         best = 'GIT';
     elseif index == 2
         best = 'HYB';
     elseif index ==3
         best = 'TSC';
     end
    fprintf('Best Model: %s\n',best);
end
%% Miscellaneous Functions
function y=rag(x)
                nu=[0.05 0.1:0.05:5];
                beta=nu;
                rho_n=beta.*exp(-psi(beta));
                B=((4.*(nu).*(gamma(nu).^2))./(pi.*(gamma(nu+.5).^2)) - 1).^(-1);
                % arithmatic mean
                % geometric mean prod(x)~(1/N)
                ma = mean(x);
                mg = geomean(x);
                p = ma/mg;
                Be = spline(rho_n ,beta ,p ); % use twice spline interpolation
                y = spline(B ,nu ,Be);
end
function [p_k,x] = kpdf(shape,scale)
%x values to span PDF over
    %x = linspace(0,10,1);    
     x = 0:0.01:5;
% Define shape and scale parameters
    c = scale; 
    v = shape;
% Define PDF distribution 
    %p_k = (sqrt(2.*v./c)./((2.^(v-1))*gamma(v))).*((sqrt((2.*v)./c).*x).^v).*besselk(v-1,sqrt((2.*v)./c).*x)
    %p_k = (2*c/gamma(v)).*(c.*x./2).^(v).*besselk(v-1,x);
    %p_k = (2*c/gamma(v)).*((0.5*c.*x).^v).*besselk(v-1,c.*x);
    p_k = (2/(c*gamma(v+1))).*((0.5*x./c).^v).*besselk(v,x./c);
%for x = 0:0.1:10
%         p_k(x) = (2*c/gamma(v))*((c*x/2)^v)*bessely(v-1,c*x)
%     end
end