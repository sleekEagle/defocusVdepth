function [] = blur_images()
%define camera parameters
f=50e-3;
px=36*1e-6;
N=1.0;  
focus=2.0;
depth_step=0.005;

rgb_dir='C:\Users\lahir\data\calibration\kinect_blur\kinect2\kinect\cameras\f_50\rgb\'
depth_dir='C:\Users\lahir\data\calibration\kinect_blur\kinect2\kinect\cameras\f_50\depth\'
out_dir='C:\Users\lahir\data\calibration\kinect_blur\kinect2\kinect\cameras\f_50\refocused\'
rgb_files=dir(rgb_dir);
depth_files=dir(depth_dir);

for i=3:(length(rgb_files))
    disp(rgb_files(i).name)
    %read image
    rgb_path=strcat(rgb_dir,rgb_files(i).name);
    depth_path=strcat(depth_dir,depth_files(i).name);
    out_path=strcat(out_dir,depth_files(i).name);
    
    rgb=double(imread(rgb_path));
    depth=imread(depth_path);
    
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
    imwrite(uint8(refocused),out_path);
end

