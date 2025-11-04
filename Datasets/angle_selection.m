function    idx = angle_selection(data_left, data_right, label, class, h5_savedir, idx)
    for id = 1 : 3
        start_index = (id - 1) * 3 + 1;
        end_index = start_index + 2;
        data_left_h = squeeze(data_left(5, start_index:end_index, :, :));
        data_left_h = permute(data_left_h, [2,3,1]);
        data_left_v = squeeze(data_left(start_index:end_index, 5, :, :));
        data_left_v = permute(data_left_v, [2,3,1]);
        data_right_h = squeeze(data_right(5, start_index:end_index, :, :));
        data_right_h = permute(data_right_h, [2,3,1]);
        data_right_v = squeeze(data_right(start_index:end_index, 5, :, :));
        data_right_v = permute(data_right_v, [2,3,1]);

        SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
        h5create(SavePath_H5_name, '/data_left_h', size(data_left_h), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/data_left_h', single(data_left_h), [1,1,1], size(data_left_h));
        h5create(SavePath_H5_name, '/data_left_v', size(data_left_v), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/data_left_v', single(data_left_v), [1,1,1], size(data_left_v));
        h5create(SavePath_H5_name, '/data_right_h', size(data_right_h), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/data_right_h', single(data_right_h), [1,1,1], size(data_right_h));
        h5create(SavePath_H5_name, '/data_right_v', size(data_right_v), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/data_right_v', single(data_right_v), [1,1,1], size(data_right_v));
        h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
        h5create(SavePath_H5_name, '/class', size(class), 'Datatype', 'single');
        h5write(SavePath_H5_name, '/class', single(class), [1,1], size(class));
        idx = idx + 1;
    end
end