clear all;
close all;
%% Global Variables
global path
global RangeBin
global DataSet
global GrazingAngle
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
%% Load Dataset
%load('TFC15_008');         % loads a variables Cdata, Info, NumOfPRIs, NumOfRangeBins, PCI, PRI_s, Waveform, lamda  
% DataSetName = 'CFA17_001';
%DataSetName = 'CFA17_002';
DataSetName = 'CFA17_001';
load(DataSetName);
DataSet = DataSetName(7:9);
%% Range Bin
RangeBin = 2;               % Specify range bin to process 
%% Path for Figures
path='C:\Users\micro\.QtWebEngineProcess\Desktop\EEE4022S\DrHerselman\';
%% Radar parameters
c = 299792458;
lambda=c/(9*(10^9));                         % Wavelength of the radar
Bandwidth_Hz = 10e6;                         % Pulsewidth = 100us, pure sinusoid at 9GHz
RangeResolution_m = c/(2*Bandwidth_Hz);      % Range resolution in meters 
if DataSet == '001'
    GrazingAngle = (1.02+1.27)/2;
elseif DataSet =='002'
     GrazingAngle = (0.472+0.525)/2;
else 
     GrazingAngle =(0.323+0.35)/2;
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
% X = Cdata(StartRangeLine:StopRangeLine,RangeBin);

% DataOneBin = Cdata(:,RangeBin); %Extract the data from the specified bin only 
% DataMeanSubtracted = DataOneBin - mean(DataOneBin);  %Normalise about mean ***CHECK*** 
% data = abs(DataMeanSubtracted)'; % amplitude of the complex clutter
%% Main

for RangeBin = 1:1:size(Cdata,2)

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
end 
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
saveas(figure(1),fullfile(path,[baseFileName '.jpeg']));
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
saveas(figure(2),fullfile(path,[baseFileName '.jpeg']));
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
saveas(figure(3),fullfile(path,[baseFileName '.jpeg']));
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
saveas(figure(4),fullfile(path,[baseFileName '.jpeg']));
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
saveas(figure(5),fullfile(path,[baseFileName '.jpeg']));
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
saveas(figure(6),fullfile(path,[baseFileName '.jpeg']));
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
%% Reflectivity Empirical Models 
% Add GIT Model, Nathonson's Table, HYB Model 
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