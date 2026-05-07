% Plot Figure 1
%
% Copyright 2025 Leibniz University Hannover, Mingzhou Yin

clc; clear; close all;
errnorm = cell(1,4); q = 5; idx = 30;
for i = 1:4
    errnorm{i} = zeros(100,3);
end

load('data/bayesSMMSmooth.mat')
errnorm{1}(:,1) = errory1;
errnorm{2}(:,1) = errory2;
errnorm{3}(:,1) = errory3;
errnorm{4}(:,1) = errory4;
load('data/bayesSMMPredict.mat')
errnorm{1}(:,2) = errory1;
errnorm{2}(:,2) = errory2;
errnorm{3}(:,2) = errory3;
errnorm{4}(:,2) = errory4;
load('data/bayesSMMControl.mat')
errnorm{1}(:,3) = sqrt(errory1)/q/sqrt(idx);
errnorm{2}(:,3) = sqrt(errory2)/q/sqrt(idx);
errnorm{3}(:,3) = sqrt(errory3)/q/sqrt(idx);
errnorm{4}(:,3) = sqrt(errory4)/q/sqrt(idx);

figure(1)
boxplotGroup(errnorm,'PrimaryLabels',{'Bayes','Approx','N4SID','Proj'}, ...
    'SecondaryLabels',{'Smoothing','Prediction','Control'}, ...
    'groupLabelType','Vertical','interGroupSpace',1,'OutlierSize',3)
fontsize(9,'points');fontname('Times New Roman');grid on;ylim([0,0.4])