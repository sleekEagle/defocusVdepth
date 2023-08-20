%define camera parameters
f=75e-3;
px=36*1e-6;
N=1.0;  
focus=2.0;
depth_step=0.005;

%read image
rgb_path='C:\Users\lahir\data\calibration\kinect_blur\kinect\blur_calib\f_75\focused\';
depth_path='C:\Users\lahir\data\calibration\kinect_blur\kinect\blur_calib\depth\';
out_path='C:\Users\lahir\data\calibration\kinect_blur\kinect\blur_calib\f_75\fdist2\';

rgb_files=dir(rgb_path);
depth_files=dir(depth_path);

for i=3:(length(rgb_files))
    rgb=double(imread(strcat(rgb_path,rgb_files(i).name)));
    depth=imread(strcat(depth_path,depth_files(i).name));
    disp(depth_files(i).name)

    %convert depth into meters
    depth=double(double(depth)/1000.0);
    %fill missing values with the mean value of depth. If not needed,
    %comment the below 2 lines
    mean_d=mean(depth(depth>0));
    depth(depth==0)=mean_d;

    d_values=(min(depth(depth>0))-depth_step):depth_step:(max(depth(depth>0))+depth_step);
    refocused = zeros(size(rgb));
    
    for k=1:length(d_values)
        if k==length(d_values)
            d=(d_values(k)+depth_step*0.5);
        else
            d=0.5*(d_values(k)+d_values(k+1));
        end
        sigma=abs(d-focus).*(1./d) / (focus-f) * f^2/N *0.3 /px;
    
        z_=zeros(size(depth));
        if k==length(d_values)
            z_((depth>=d_values(k)))=1;
        else
            z_((depth<d_values(k+1)) & (depth>=d_values(k)))=1;
        end
        %dialate z_ hopefully to cover zinvalid depth areas
        z_=repmat(z_,1,1,3);
        img_=rgb.*z_;
        refocused_=imgaussfilt(img_,sigma+0.0001);
        refocused=refocused+refocused_;
    end
    imwrite(uint8(refocused),strcat(out_path,depth_files(i).name));

end
