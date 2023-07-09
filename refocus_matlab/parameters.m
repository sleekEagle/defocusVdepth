% optical parameter of the simulated lens (in m)
px=36*1e-6; % x2 car les images sont sous-échantillonnées 
%px=1*1e-2;
N=2.0;  
f=3.9*1e-3; %2.9*1e-3 for kinect
mode_='gaussian';

% minimal step of depth values in the depth map
step_depth=0.005;

% min value to filter dark images
min_mean = 20;

focus=0.05;

max_depth = 10.0; % Even though max_depth from xtion is 3.5