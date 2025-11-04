clc;close all;clear;

dataset_path = 'D:\LFIQA_Datasets\Stitched_WLFI_Database_NBU\MLI version\'; % Set the dataset path here
savepath = '.\WLFI_224x224_dual\'; % Set the save path here

% for WLFI_Database_NBU
load('WLFI_NBU_info.mat');
load('Stitched_WLFI_Database_MOS.mat');

Distorted_sceneNum = 320;

angRes = 9;             
patchsize = 224;         
stride = 224; 

inum = 1;
for iScene = 1 : Distorted_sceneNum
    
    tic;
    idx = 1;
    h5_savedir = [savepath, '\', WLFI_NBU_info{1}{iScene}, '\',  WLFI_NBU_info{2}{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, WLFI_NBU_info{1}{iScene}, '_',  WLFI_NBU_info{2}{iScene}, '.bmp'];
    LF = imread(dataPath);
    [H_raw, W_raw, ~] = size(LF);

    LF = permute(reshape(LF,[9, H_raw/9, 9, W_raw/9, 3]),[1,3,2,4,5]);
    [U, V, ~, ~, ~] = size(LF);
    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, :);
    [U, V, H, W, ~] = size(LF);

    LF_left = LF(:, :, :, 1:ceil(W/2), :);
    [~, ~, H, W_left, ~] = size(LF_left);

    LF_right = LF(:, :, :, ceil(W/2)+1:end, :);
    [~, ~, H, W_right, ~] = size(LF_right);
    
    label = Stitched_WLFI_Database_MOS(iScene);
    class = str2double(WLFI_NBU_info{2}{iScene}(end));
    
    sequential_manner = 1;
    while sequential_manner <=4
        switch sequential_manner
            case 1
                for h = 1 : stride : H - patchsize + 1
                    for w = 1 : stride : W_right - patchsize + 1
                        data_left = single(zeros(U, V, patchsize, patchsize));
                        data_right = single(zeros(U, V, patchsize, patchsize));
                        for u = 1 : U
                            for v = 1 : V                        
                                patch_left = squeeze(LF_left(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                                patch_left = rgb2ycbcr(patch_left);
                                patch_left = squeeze(patch_left(:,:,1));
                                data_left(u, v, :, :) = patch_left;
                                patch_right = squeeze(LF_right(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                                patch_right = rgb2ycbcr(patch_right);
                                patch_right = squeeze(patch_right(:,:,1));
                                data_right(u, v, :, :) = patch_right;
                            end
                        end
                        idx = angle_selection(data_left, data_right, label, class, h5_savedir, idx);
                    end
                end
                sequential_manner = sequential_manner + 1;
            case 2
                for h = 1 : stride : H - patchsize + 1
                    for w = W_right : -stride : patchsize
                        data_left = single(zeros(U, V, patchsize, patchsize));
                        data_right = single(zeros(U, V, patchsize, patchsize));
                        for u = 1 : U
                            for v = 1 : V                        
                                patch_left = squeeze(LF_left(u, v, h : h+patchsize-1, w-patchsize+1 : w, :));
                                patch_left = rgb2ycbcr(patch_left);
                                patch_left = squeeze(patch_left(:,:,1));
                                data_left(u, v, :, :) = patch_left;
                                patch_right = squeeze(LF_right(u, v, h : h+patchsize-1, w-patchsize+1 : w, :));
                                patch_right = rgb2ycbcr(patch_right);
                                patch_right = squeeze(patch_right(:,:,1));
                                data_right(u, v, :, :) = patch_right;
                            end
                        end
                        idx = angle_selection(data_left, data_right, label, class, h5_savedir, idx);
                    end
                end
                sequential_manner = sequential_manner + 1;
            case 3
                for h = H : -stride : patchsize
                    for w = 1 : stride : W_right - patchsize + 1
                        data_left = single(zeros(U, V, patchsize, patchsize));
                        data_right = single(zeros(U, V, patchsize, patchsize));
                        for u = 1 : U
                            for v = 1 : V                        
                                patch_left = squeeze(LF_left(u, v, h-patchsize+1 : h, w : w+patchsize-1, :));
                                patch_left = rgb2ycbcr(patch_left);
                                patch_left = squeeze(patch_left(:,:,1));
                                data_left(u, v, :, :) = patch_left;
                                patch_right = squeeze(LF_right(u, v, h-patchsize+1 : h, w : w+patchsize-1, :));
                                patch_right = rgb2ycbcr(patch_right);
                                patch_right = squeeze(patch_right(:,:,1));
                                data_right(u, v, :, :) = patch_right;
                            end
                        end
                        idx = angle_selection(data_left, data_right, label, class, h5_savedir, idx);
                    end
                end
                sequential_manner = sequential_manner + 1;
            case 4
                for h = H : -stride : patchsize
                    for w = W_right : -stride : patchsize
                        data_left = single(zeros(U, V, patchsize, patchsize));
                        data_right = single(zeros(U, V, patchsize, patchsize));
                        for u = 1 : U
                            for v = 1 : V                        
                                patch_left = squeeze(LF_left(u, v, h-patchsize+1 : h, w-patchsize+1 : w, :));
                                patch_left = rgb2ycbcr(patch_left);
                                patch_left = squeeze(patch_left(:,:,1));
                                data_left(u, v, :, :) = patch_left;
                                patch_right = squeeze(LF_right(u, v, h-patchsize+1 : h, w-patchsize+1 : w, :));
                                patch_right = rgb2ycbcr(patch_right);
                                patch_right = squeeze(patch_right(:,:,1));
                                data_right(u, v, :, :) = patch_right;
                            end
                        end
                        idx = angle_selection(data_left, data_right, label, class, h5_savedir, idx);
                    end
                end
                sequential_manner = sequential_manner + 1;
        end
    end
    disp(['第 ', num2str(inum), ' 个场景生成', '运行时间: ',num2str(sprintf('%.3f', toc))]);
    inum = inum + 1;
end




