clear;
clc;

SUBS = 1:2;

for sub = SUBS

    clearvars -except sub SUBS
    fprintf('sub:%d\n',sub);
    pd = 'D:\Internship_INSA\Haxby\Classification_subjectwise';
    if ~exist(pd)
        mkdir(pd)
    end

    wd = sprintf('D:\\Internship_INSA\\Haxby\\sub_%d',sub);
    cd(wd)

    save_dir = sprintf('D:\\Internship_INSA\\Haxby\\sub_%d\\',sub);
    if ~exist(save_dir)
        mkdir(save_dir)
    end

    win =8;
    load(sprintf('EpochedDataLabelsShifted_%dvols.mat',win));
    
    % first concatenate trials 
    ts_concat_2 = []; ts_concat_4 = [];
    labels_concat = [];
    vol_concat = [];
    trial_concat = [];
    ts_avg_concat_2 = []; ts_avg_concat_4 = [];
    % ts_avg_new = [];
    
    for rn = includeBlocks
    
        ts_concat_2 = [ts_concat_2;ts_2_mat(:,:,rn)];
        ts_avg_concat_2 = [ts_avg_concat_2;ts_2_avg(:,:,rn)]; 
        ts_concat_4 = [ts_concat_4;ts_4_mat(:,:,rn)];
        ts_avg_concat_4 = [ts_avg_concat_4;ts_4_avg(:,:,rn)]; 
        labels_concat = [labels_concat,labels(rn,:)];
        vol_concat = [vol_concat,vol_ind(rn,:)];
        trial_concat = [trial_concat,trial_label(rn,:)];
        
%         for trials = 1:8
%             ts_avg_new((rn-1)*8+trials,:) = mean(ts_concat_2(((rn-1)*64+(trials-1)*8+1):((rn-1)*64+(trials-1)*8+8),:));
%         end
  
    end
    
    concat_acc_2 = []; concat_acc_4 = [];
    cnt = 0;
    unique_labels = unique(labels_concat);
    
    for element_no = numel(unique_labels)
        for loo = unique(trial_concat)
            binary_label = strcmp(labels_concat,unique_labels(element_no));
            cnt = cnt + 1;
            fprintf('win %d, loo %d/%d\n',win,cnt,length(unique(trial_concat)));
            ind_diag_in{cnt} = find(trial_concat ~= loo);
            ind_diag_out{cnt} = find(trial_concat == loo);

            [pred_2{cnt},acc_2{cnt}] = Model_two(ts_concat_2(ind_diag_in{cnt},:),binary_label(:,ind_diag_in{cnt}),true,ts_concat_2(ind_diag_out{cnt},:),binary_label(:,ind_diag_out{cnt}));
            [pred_4{cnt},acc_4{cnt}] = Model_two(ts_concat_4(ind_diag_in{cnt},:),binary_label(:,ind_diag_in{cnt}),true,ts_concat_4(ind_diag_out{cnt},:),binary_label(:,ind_diag_out{cnt}));

            concat_acc_2 = [concat_acc_2;acc_2{cnt}];
            concat_acc_4 = [concat_acc_4;acc_4{cnt}];
        end
    end
    
    %%
    A_2(element_no) = sum(concat_acc_2)/size(concat_acc_2,1);
    A_4(element_no) = sum(concat_acc_4)/size(concat_acc_4,1);

    %%
    filename = sprintf('%sClassification_subjectwise_binary.mat',save_dir);

    save (filename);

end

%% functions for classification 

function [Mdl,ce,acc] = CVModel_two(X,Y,k,std)
    Mdl = fitcsvm(X,Y,'KFold',k,'Standardize',std);
    ce = kfoldLoss(Mdl);
    acc = 1 - ce;
end

function [CVMdl,ce,acc] = CVModel_multi(X,Y,k,std)
    t = templateSVM('Standardize',std);
    CVMdl = fitcecoc(X,Y,'Learners',t,'Kfold',k);
    ce = kfoldLoss(CVMdl);
    acc = 1 - ce;
end

function [pred_label,acc] = Model_two(X,Y,std,x_pred,y_pred)
    Mdl = fitcsvm(X,Y,'Standardize',std);
    pred_label = predict(Mdl,x_pred);
    acc = (pred_label == y_pred');
end

function [pred_label,acc] = Model_multi(X,Y,std,x_pred,y_pred)
    t = templateSVM('Standardize',std);
    Mdl = fitcecoc(X,Y,'Learners',t);
    pred_label = predict(Mdl,x_pred);
    y_pred = y_pred';
    for ii = 1:size(y_pred,1)
        acc(ii,1) = isequal(pred_label(ii), y_pred(ii));
    end
end
