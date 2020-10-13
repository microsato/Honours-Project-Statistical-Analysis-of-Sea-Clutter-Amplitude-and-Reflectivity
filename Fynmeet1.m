
% Plot spectrogram of a specific range bin 

clear all;
close all;

load('TFC15_008');  % loads a variables Cdata, Info, NumOfPRIs, NumOfRangeBins, PCI, PRI_s, Waveform, lamda  

%% Radar parameters
c = 299792458;
lambda=c/(9*(10^9));                         % Wavelength of the radar
Bandwidth_Hz = 10e6;                         % Pulsewidth = 100us, pure sinusoid at 9GHz
RangeResolution_m = c/(2*Bandwidth_Hz);      % Range resolution in meters 

%% Plot Cdata 
PRI = PRI_s; % TFC15_008
% VRange_m = (1:1:size(Cdata,2)); 
% VRangeBins = 1:1:size(Cdata,2);
% TimeVector = (1:1:size(Cdata,1))*PRI;
% figure; imagesc(VRange_m,TimeVector, 20*log10(abs(Cdata)));
% xlabel('Range (bin)'); ylabel('Time (s)');  colormap('jet'); colorbar;

%% Extract one range bin 
RangeBin = 79;  % Specify range bin to process 
fs = 1/PRI;
ts = 1/fs;
NumRangeLines = size(Cdata,1);

StartRangeLine = round(NumRangeLines/2); 
StopRangeLine = round(NumRangeLines/4*3);
X = Cdata(StartRangeLine:StopRangeLine,RangeBin);

%% Generate and plot histogram

% Extract one range bin of data
RangeBin = 79;  % Specify range bin to process 
DataOneBin = Cdata(:,RangeBin); %Extract the data from the specified bin only 
DataMeanSubtracted = DataOneBin - mean(DataOneBin);  %Normalise about mean ***CHECK*** 
y = abs(DataMeanSubtracted).'; % amplitude of the complex clutter

% Plot histogram
sizeData = length(y);
N = sizeData;
binSize = 1000;
data = sort(y,'ascend');
figure()
h = histogram(data,100,'Normalization','pdf');
hold on;
x = 0:0.01:data(sizeData)+1;

%% Rayleigh Distribution 
%Plot PDF of MoM and MLE estimates
[p_mle_rayl,p_mom_rayl,sigma_mle_rayl,sigma_mom_rayl] = rayleighPDF(data);
plot(x,p_mle_rayl,'Linewidth',1);
hold on;
plot(x,p_mom_rayl,'Linewidth',1);
hold on;
%Goodness of Fit: Rayleigh
[modChiSqr_rayl_mle, modChiSqr_rayl_mom] = modChiSqr_rayl(10,0.1,data,sigma_mle_rayl,sigma_mom_rayl); 
%% Lognormal Distribution
[p_mle_logn,p_mom_logn,sigma_mle_logn,sigma_mom_logn,mu_mle_logn,mu_mom_logn] = lognPDF(data);
plot(x,p_mle_logn,'Linewidth',1);
hold on;
plot(x,p_mom_logn,'Linewidth',1);
hold on;
%Goodness of Fit: LogNormal
[modChiSqr_logn_mle, modChiSqr_logn_mom] = modChiSqr_logn(10,0.1,data,sigma_mle_logn,sigma_mom_logn,mu_mle_logn,mu_mom_logn); 

%% Weibull Distribution 
[p_mle_wbl,p_mom_wbl,shape_mle_wbl,shape_mom_wbl,scale_mle_wbl,scale_mom_wbl] = wblPDF(data);
plot(x,p_mle_wbl,'Linewidth',1);
hold on;
plot(x,p_mom_wbl,'Linewidth',1);
hold on;
%Goodness of Fit: LogNormal
[modChiSqr_wbl_mle, modChiSqr_wbl_mom] = modChiSqr_wbl(10,0.1,data,shape_mle_wbl,shape_mom_wbl,scale_mle_wbl,scale_mom_wbl); 

%% K-Dsitribution
[p_watts_k,p_mom_k,p_r_k,shape_watts_k,shape_mom_k,shape_r_k,scale_watts_k,scale_mom_k,scale_r_k] = kPDF(data);
plot(x,p_watts_k,'Linewidth',1);
hold on;
plot(x,p_mom_k,'Linewidth',1);
hold on;
plot(x,p_r_k,'Linewidth',1);
%Goodness of Fit: K-Dsitribution
%[modChiSqr_k_watts, modChiSqr_k_mom, modChiSqr_k_r] = modChiSqr_k(10,0.1,data,shape_watts_k,shape_mom_k,shape_r_k,scale_watts_k,scale_mom_k,scale_r_k);
K = 10;
PFA = 0.1;
N=length(data);       %get length of sea clutter data
    CDF_start=      1-PFA;   %Start at this probability in the CDF function (translates to PFA in PDF) 
    stepSize =      PFA/K;
    probability_intervals=  CDF_start:stepSize:1; 
    %intervals_CDF=  0:0.1:1; 
    x = 0:1e-4:data(N); %Get cdf of k distribution over the desired range
    k_cdf_watts= 1 - ((2/gamma(shape_watts_k))*((scale_watts_k.*x./2).^shape_watts_k).*besselk(shape_watts_k,scale_watts_k.*x)); %You dont need to change this
    k_cdf_mom= 1 - ((2/gamma(shape_mom_k))*((scale_mom_k.*x./2).^shape_mom_k).*besselk(shape_mom_k,scale_mom_k.*x)); %You dont need to change this
    k_cdf_r= 1 - ((2/gamma(shape_r_k))*((scale_r_k.*x./2).^shape_r_k).*besselk(shape_r_k,scale_r_k.*x)); %You dont need to change this
    
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
    modChiSqr_k_watts= sum((fi_watts-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_k_mom= sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_k_r= sum((fi_r-(PFA)*N/K).^2/((PFA)*N/K));

%% PLOT
legend('Rayleigh MLE','Rayleigh MoM','Lognormal MLE','Lognormal MoM','Weibull MLE','Weibull MoM','K Watts','K MoM','K Rag');
fprintf('Modified Chi-Squared Test: \nRayleigh MLE: %f \nRayleigh MoM: %f \nLognormal MLE: %f \nLognormal MoM: %f \nWeibull MLE: %f \nWeibull MoM: %f \nK Watts: %f \nK MoM: %f \nK Rag: %f \n',modChiSqr_rayl_mle, modChiSqr_rayl_mom,modChiSqr_logn_mle, modChiSqr_logn_mom,modChiSqr_wbl_mle, modChiSqr_wbl_mom,modChiSqr_k_watts, modChiSqr_k_mom, modChiSqr_k_r);
function [p_mle,p_mom,sigma_mle,sigma_mom] = rayleighPDF(data)
    N = numel(data);
    
    %MLE: Get MLE sigma estimate
    sigma_mle = raylfit(data);
    %MoM: Get MoM sigma estimate
    sigma_mom = mean(data)*sqrt(2)/sqrt(pi); %From https://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_rayleigh_theory.html
    %Get PDF Function
    x = 0:0.01:data(N)+1;
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
    x = 0:0.01:data(N)+1;
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
    x = 0:0.01:data(N)+1;
    p_mle = (shape_mle/scale_mle).*(x./scale_mle).^(shape_mle-1).*exp(-(x./scale_mle).^shape_mle);
    p_mom = (shape_mom/scale_mom).*(x./scale_mom).^(shape_mom-1).*exp(-(x./scale_mom).^shape_mom);
end

function [p_watts,p_mom,p_r,shape_watts,shape_mom,shape_r,scale_watts,scale_mom,scale_r] = kPDF(data)
    N = numel(data);
    
    %Watts's Method: Get shape and scale estimate using 2nd and 4th moments
    m2 = (1/N)*sum(data.^2); %second sample moment
    m4 = (1/N)*sum(data.^4); %fourth sample moment
    
    shape_watts = ((m4/(2*(m2)^2))-1)^(-1);
    scale_watts = 2*sqrt(shape_watts/m2);

    %MoM: Get shape and scale estimate using 1st and 2nd moments
    m1 = mean(data);        %first sample moment
    m2 = mean(data.^2);     %second sample moment
    
    fun = @(v)(4.*v.*gamma(v).^2)./(pi.*gamma(v+0.5).^2) -m2./(m1.^2);
    shape_mom= fzero(fun,shape_watts);
    scale_mom = gamma(shape_mom+0.5)*sqrt(pi)/(gamma(shape_mom)*m1);
    
    %Raghavan's Method
    shape_r = rag(data);
    scale_r = (2/mean(data))*(gamma(shape_r+0.5)*gamma(1.5))/(gamma(shape_r));
    
    %Get PDF Function
    x = 0:0.01:data(N)+1;
    p_watts= (2*scale_watts/gamma(shape_watts)).*((0.5*scale_watts.*x).^shape_watts).*besselk(shape_watts-1,scale_watts.*x);
    p_mom= (2*scale_mom/gamma(shape_mom)).*((0.5*scale_mom.*x).^shape_mom).*besselk(shape_mom-1,scale_mom.*x);
    p_r= (2*scale_r/gamma(shape_r)).*((0.5*scale_r.*x).*shape_r).*besselk(shape_r-1,scale_r.*x);
end
%% Modified Chi-Squared Test Functions
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
    modChiSqr_mle= sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_mom= sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K));
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
    modChiSqr_mle= sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_mom= sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K));
    
   

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
    modChiSqr_mle= sum((fi_mle-(PFA)*N/K).^2/((PFA)*N/K));
    modChiSqr_mom= sum((fi_mom-(PFA)*N/K).^2/((PFA)*N/K));

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