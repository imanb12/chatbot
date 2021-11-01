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
    
    for rn = includeBlocks
    
        ts_concat_2 = [ts_concat_2;ts_2_mat(:,:,rn)];
        ts_avg_concat_2 = [ts_avg_concat_2;ts_2_avg(:,:,rn)]; 
        ts_concat_4 = [ts_concat_4;ts_4_mat(:,:,rn)];
        ts_avg_concat_4 = [ts_avg_concat_4;ts_4_avg(:,:,rn)]; 
        labels_concat = [labels_concat,labels(rn,:)];
        vol_concat = [vol_concat,vol_ind(rn,:)];
        trial_concat = [trial_concat,trial_label(rn,:)];
  
    end
    
    
    %% Classification with all features
    
    concat_acc_2 = []; concat_acc_4 = [];
    cnt = 0;
    num_trial = numel(unique(trial_concat));
    [c,total_cnt] = cvpart_haxby(unique(trial_concat),true,0);
    
    for loo = unique(trial_concat)
        cnt = cnt + 1;
        fprintf('win %d, loo %d/%d\n',win,cnt,total_cnt);
        [ind_diag_in{cnt},ind_diag_out{cnt}] = cvpart_volwise(c,cnt,trial_concat);

        [pred_2{cnt},acc_2{cnt}] = Model_multi(ts_concat_2(ind_diag_in{cnt},:),labels_concat(:,ind_diag_in{cnt}),true,ts_concat_2(ind_diag_out{cnt},:),labels_concat(:,ind_diag_out{cnt}));
        [pred_4{cnt},acc_4{cnt}] = Model_multi(ts_concat_4(ind_diag_in{cnt},:),labels_concat(:,ind_diag_in{cnt}),true,ts_concat_4(ind_diag_out{cnt},:),labels_concat(:,ind_diag_out{cnt}));

        concat_acc_2 = [concat_acc_2;acc_2{cnt}];
        concat_acc_4 = [concat_acc_4;acc_4{cnt}];
    end
    
    %%
    A_2 = sum(concat_acc_2)/size(concat_acc_2,1);
    A_4 = sum(concat_acc_4)/size(concat_acc_4,1);
    

    %% Best feature selection
    fs_2 = []; fs_4 =[];
    num_of_best_features = 5;
    num_features = size(ts_concat_2,2);
    opts = statset('Display','iter');
    trials = (1:num_trial)';
    feature_count = repmat((1:num_features),num_trial,1);
    c = cvpartition(unique(trial_concat),'KFold',6);
    
    classf =  @(XT,yT,Xt,yt)multi_error(XT, ts_concat_2, labels_concat, yT, yt, true, trial_concat);
    [fs_2(sub,:),~] = sequentialfs(classf,feature_count,trials,'cv',c,'nfeatures',num_of_best_features,'options',opts);
    
    classf =  @(XT,yT,Xt,yt)multi_error(XT, ts_concat_4, labels_concat, yT, yt, true, trial_concat);
    [fs_4(sub,:),~] = sequentialfs(classf,feature_count,trials,'cv',c,'nfeatures',num_of_best_features,'options',opts);

    
    %% Classification with best features
    cnt = 0;
    
    for loo = unique(trial_concat)
        cnt = cnt + 1;
        fprintf('win %d, loo %d/%d\n',win,cnt,length(unique(trial_concat)));
        [ind_diag_in{cnt},ind_diag_out{cnt}] = cvpart_volwise(c,cnt,trial_concat);

        [pred_2{cnt},acc_2{cnt}] = Model_multi(ts_concat_2(ind_diag_in{cnt},fs_2(sub,:)),labels_concat(:,ind_diag_in{cnt}),true,ts_concat_2(ind_diag_out{cnt},fs_2(sub,:)),labels_concat(:,ind_diag_out{cnt}));
        [pred_4{cnt},acc_4{cnt}] = Model_multi(ts_concat_4(ind_diag_in{cnt},fs_4(sub,:)),labels_concat(:,ind_diag_in{cnt}),true,ts_concat_4(ind_diag_out{cnt},fs_4(sub,:)),labels_concat(:,ind_diag_out{cnt}));

        concat_acc_2 = [concat_acc_2;acc_2{cnt}];
        concat_acc_4 = [concat_acc_4;acc_4{cnt}];
    end
    
    %%
    A_2_best = sum(concat_acc_2)/size(concat_acc_2,1);
    A_4_best = sum(concat_acc_4)/size(concat_acc_4,1);
    
    
    filename = sprintf('%sClassification_subjectwise_multi.mat',save_dir);

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

function [c,total_cnt] = cvpart_haxby(y,leaveout,k)
    if(leaveout)
        c = cvpartition(y,'LeaveOut');
        total_cnt = size(y,2);
    else
        c = cvpartition(y,'KFold',k);
        total_cnt = k;
    end
end

function [data_in, data_out] = cvpart_volwise(c,cnt,trials)
    
    train_trial = find(training(c,cnt)==1);
    test_trial = find(test(c,cnt)==1);
    data_in = [];
    for ii = 1:size(train_trial,1)
        data_in = [data_in; find(trials == train_trial(ii))];
    end
    data_out = [];
    for ii = 1:size(test_trial,1)
        data_out = [data_out; find(trials == test_trial(ii))];
    end
end

function error = multi_error(fts,X,Y,train,test,std,trials)
%     disp(fts);
    ind_diag_in = [];
    for ii = 1:size(train,1)
        ind_diag_in = [ind_diag_in; find(trials == train(ii))];
    end
    ind_diag_out = [];
    for ii = 1:size(test,1)
        ind_diag_out = [ind_diag_out; find(trials == test(ii))];
    end
    X_train = X(ind_diag_in,fts(1,:));
    x_pred = X(ind_diag_out,fts(1,:));
    Y_train = Y(:,ind_diag_in);
    y_pred = Y(:,ind_diag_out);
    
    t = templateSVM('Standardize',std);
    Mdl = fitcecoc(X_train,Y_train,'Learners',t);
    pred_label = predict(Mdl,x_pred);
    y_pred = y_pred';
    error = 0;
    for ii = 1:size(y_pred,1)
        error = error + (~isequal(pred_label(ii), y_pred(ii)));
    end
    error = error/size(y_pred,1);
end
