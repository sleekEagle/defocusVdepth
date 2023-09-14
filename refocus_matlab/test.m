%read image
rgb=double(imread('C:\Users\lahir\data\kinectmobile\kinect\rgb\84.jpg'));
depth=imread('C:\Users\lahir\data\kinectmobile\kinect\depth\84.png');
%convert depth into meters
depth=double(depth)/(1000.0);

%rgb=rgb(500:800,500:800,:);
%depth=depth(500:800,500:800,:);


%define camera parameters
f=40e-3;
px=36*1e-6;
N=1.0;  
focus=2.0;
n_sigmas=4;

refocused=double(zeros(size(rgb)));
sigma=abs(depth-focus).*(1./depth) / (focus-f) * f^2/N *0.3 /px ;

values=zeros(size(rgb));
tic
for i=1:size(rgb,1)
    for j=1:size(rgb,2)
        %we do not process missing depth values
        if depth(i,j)==0
            refocused(i,j,:)=refocused(i,j,:)+rgb(i,j);
            values(i,j,:)=values(i,j,:)+1;
            continue
        %else do defocus blurring...
        else
            s=sigma(i,j);
        end  
        min_i=int16(max(1,i-n_sigmas*s));
        max_i=int16(min(size(rgb,1),i+n_sigmas*s));
        min_j=int16(max(1,j-n_sigmas*s));
        max_j=int16(min(size(rgb,2),j+n_sigmas*s));
        img_=rgb(min_i:max_i,min_j:max_j,:);
        z_=zeros(size(img_));
        z_(uint16(size(img_,1)/2),uint16(size(img_,2)/2),:)=1;
        img_=img_.*z_;
        refocused_=imgaussfilt(img_,s+0.0001);
        refocused(min_i:max_i,min_j:max_j,:)=refocused(min_i:max_i,min_j:max_j,:)+refocused_;
        values(min_i:max_i,min_j:max_j,:)=values(min_i:max_i,min_j:max_j,:)+1;
    end
    disp(i);
    %imshow(uint8(refocused));
end
toc

refocused_scaled=refocused./values;
refocused_scaled=refocused_scaled./ max(refocused_scaled,[],'all');
refocused_scaled=refocused_scaled*255;

imshow(uint8(refocused_scaled));

montage({uint8(rgb),uint8(refocused)})






