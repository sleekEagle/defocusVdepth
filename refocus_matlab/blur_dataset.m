function [] = blur_dataset()
    %  BLUR_DATASET generates synthetically blurred images when given pairs of RGB 
    %   and corresponding depth maps. This code is an adaptation of the layered
    %   approach proposed in "A layer-based restoration framework for variable 
    %   aperture photography", Hasinoff, S.W., Kutulakos, K.N., IEEE 11th 
    %   International Conference on Computer Vision, to create a realistic defocus
    %   blur:

    % Authors: Pauline Trouvé-Peloux and Marcela Carvalho.
    % Year: 2017

    % load parameters
    % Uncomment the next line to reproduce experiments from section 3 of the 
    % paper. Change focus on the file to make different tests.
    parameters;
    % Uncomment the next line to reproduce the pre-training dataset from section
    % 4 of the paper.
    % parameters_DFD_indoor;

    h = waitbar(0,'Initializing waitbar...');
    s = clock;
    
    %path_rgb = ['C:\Users\lahir\kinect_hand_data\extracted\lahiru1\cropped\rgb\'];
    %path_depth = ['C:\Users\lahir\kinect_hand_data\extracted\lahiru1\cropped\depth\'];

    path_rgb = ["C:\Users\lahir\data\calibration\kinect_blur\kinect\blur_calib\rgb\"];
    path_depth = ["C:\Users\lahir\data\matlabtest\depth\"];
    dest_path_rgb = ["C:\Users\lahir\data\matlabtest\"];
    
    for j=1:length(path_rgb)
        source_rgb=path_rgb(j);
        source_depth=path_depth(j);
        dest_rgb=dest_path_rgb(j);

        create_dir(dest_rgb);
        
        
        contents_rgb = dir(source_rgb);
        contents_depth = dir(source_depth);

        for i=1:(length(contents_rgb)-2)
            disp('contents rgb:')
            disp(contents_rgb(i+2))
            if(rem(i-1,100) == 0)
                s=clock;
            end
            disp(contents_rgb(i+2).name)
            
            % read images
            im=double(imread(source_rgb+contents_rgb(i+2).name));
            %if depth data is in .exr format
            %depth=(exrread([path_depth contents_depth(i+2).name]));
            %if depth data is in .png format
            disp(contents_depth(i+2).name)
            depth=(imread(source_depth+contents_depth(i+2).name));
            
            %depth=depth(:,:,1);
            
            %conversion into depth values in meters
            depth=double(depth)/(1000.0);
    
            [im_refoc, ~, ~, D]=refoc_image(im,depth,step_depth,focus,f,N,px,mode_);
            %fname_str=contents_rgb(i+2).name;
            fname_str=strcat('f_',string(f),'.jpg');
            %[m,n]=size(fname_str);
            %snum=str2num(fname_str(1:n-4));
            %fname=num2str(snum,'%06.f')+"_01rgb.png";
            fname=append(dest_rgb,fname_str)
            imwrite(uint8(im_refoc), fname)
            %imwrite(uint16(depth*1000), [dest_path_depth contents_depth(i+2).name]) % save depth in milimeters   
            if (rem(i-1,100) == 0)
                is = etime(clock, s);
                esttime = is * (length(contents_rgb)-2 -i);
            end
    
            [hours, min, sec] = sec_hms(esttime - etime(clock,s));
    
            perc = i/(length(contents_rgb)-2);
            waitbar(perc,h,...
                [' focus: ' num2str(focus) ' ' sprintf('%3.1f%% [%2.0fh%2.0fm%2.0fs]',...
                perc*100, hours, min, sec)]);

        end
    end

%       figure
%       plot(D, sigma_vec)

end

function [] = create_dir(dir_path)
    if(exist(dir_path)~=7)
        mkdir(dir_path)
    else
        display([dir_path  'already exists'])
    end
end