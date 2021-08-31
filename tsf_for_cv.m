% get the tsf data for cross validation
% need to install biosig to extract .gdf or change to extract from the .mat
function data = tsf_for_cv(subject_index)
% need biosig-code toolbox
% the function to extract the data of one session 
% change NaN to 0
% filter to  4-40 Hz

%% BioSig Get the data 
% T data
% subject_index = 2; %1-9
session_type = 'T';

dir = ['D:\MI\MI\BCIData\Adata\A0',num2str(subject_index),session_type,'.gdf'];
[s, HDR] = sload(dir);

% Label 
% label = HDR.Classlabel;
labeldir = ['D:\MI\MI\BCIData\Alabel\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir);
label_1 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;


% E data tsf
session_type = 'E';
dir = ['D:\MI\MI\BCIData\Adata\A0',num2str(subject_index),session_type,'.gdf'];
% dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
[s, HDR] = sload(dir);

% Label 
% label = HDR.Classlabel;
labeldir = ['D:\MI\MI\BCIData\Alabel\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir);
label_2 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_2 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_2(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_2(isnan(data_2)) = 0;


%% shuffle and split the data 
data = cat(3,data_1,data_2);
label = [label_1;label_2];
label = label-1;
% num = size(data, 3);
pindex = randperm(576);
data = data(:,:,pindex);
label = label(pindex);

%% You can choose the size of the test set according to your habits
e_data = data(:,:,1:60);
e_label = label(1:60);
t_data = data(:,:,61:576);
t_label = label(61:576);

%% calculate the tsf of training data
index_0 = find(t_label==0);
index_1 = find(t_label==1);
index_2 = find(t_label==2);
index_3 = find(t_label==3);


%% TSF 
% 4-40 Hz
fc = 250;
fb_data = zeros(1000,22,516);

Wl = 4; Wh = 40; % Í¨´ø·¶Î§
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:516
    fb_data(:,:,j) = filtfilt(b,a,t_data(:,:,j));
end


eeg_mean = mean(fb_data,3);
eeg_std = std(fb_data,1,3); 
fb_data = (fb_data-eeg_mean)./eeg_std;


%% obtain W of each band
W = zeros(22,22,4);
for nclass = 0:3 % let L represent one class, R represent other three classes

    if nclass == 0 
        index_L = index_0;
        index_R = [index_1;index_2;index_3];
    elseif nclass == 1
        index_L = index_1;
        index_R = [index_0;index_2;index_3];
    elseif nclass == 2
        index_L = index_2;
        index_R = [index_0;index_1;index_3];
    elseif nclass == 3
        index_L = index_3;
        index_R = [index_0;index_1;index_2];
    end   
    index_R = sort(index_R);
     
    Cov_L = zeros(22,22,length(index_L));
    Cov_R = zeros(22,22,length(index_R));
    for nL = 1:length(index_L)
        E = fb_data(:,:,index_L(nL));
        E = E'; % channel*sample point, don't mind
        EE = E*E';
        Cov_L(:,:,nL) = EE./trace(EE);
     end
     for nR = 1:length(index_R)
     E = fb_data(:,:,index_R(nR));
     E = E';
     EE = E*E';
     Cov_R(:,:,nR) = EE./trace(EE);
     end
     CovL = mean(Cov_L,3);
     CovR = mean(Cov_R,3);
     CovTotal = CovL + CovR;
           
     [Uc,lambda] = eig(CovTotal); % Uc is the eigenvector matrix, Dt is the diagonal matrix of eigenvalue
     eigenvalues = diag(lambda);
     [eigenvalues,egIndex] = sort(eigenvalues, 'descend');
     Ut = Uc(:,egIndex); % sort as the descend order
        
     P = sqrt(diag(eigenvalues)^-1)*Ut';
        
     SL = P*CovL*P';
     SR = P*CovR*P';   
                
     [BL,lambda_L] = eig(SL);
     evL = diag(lambda_L);
     [evL,egI] = sort(evL, 'descend');
     B = BL(:,egI);
     % [BR,lambda_R] = eig(SR);
     % evR = diag(lambda_R);
     % evR = sort(evR);
     w = P'*B;
     W(:,:,nclass+1) = w;
end
% Use the first two and the last two
% W1 = W(:,[1:2 (end-1):end],1);
% W2 = W(:,[1:2 (end-1):end],2);
% W3 = W(:,[1:2 (end-1):end],3);
% W4 = W(:,[1:2 (end-1):end],4);

% use the first four          
W1 = W(:,1:4,1);
W2 = W(:,1:4,2);
W3 = W(:,1:4,3);
W4 = W(:,1:4,4);

Wb = [W1,W2,W3,W4]; % Z = W' * X

%% Training data tsf filtered
data = zeros(1000,16,516);
for ntrial = 1:516
    tdata = fb_data(:,:,ntrial);
    tdata = tdata';
    tdata = Wb'*tdata;
    data(:,:,ntrial) = tdata';
end
% kk = 2 / (max(data,[],'all') - min(data,[],'all'));
% data = -1 + kk * (data - min(data,[],'all'));

% data = data/8;
% data = tanh(data);
saveDir = ['D:\MI\MI\BCIData\tsf_cv_data\A0',num2str(subject_index),'T.mat'];
label = t_label + 1;
save(saveDir,'data','label');


%% Test data tsf filtered
% 4-40 Hz
fc = 250;
fb_data = zeros(1000,22,60);

Wl = 4; Wh = 40; % Í¨´ø·¶Î§
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:60
    fb_data(:,:,j) = filtfilt(b,a,e_data(:,:,j));
end

fb_data = (fb_data-eeg_mean)./eeg_std;

data = zeros(1000,16,60);

for ntrial = 1:60
    edata = fb_data(:,:,ntrial);
    edata = edata';
    edata = Wb'*edata;
    data(:,:,ntrial) = edata';
end

% data = data/8;
% data = tanh(data);
label = e_label+1;
saveDir = ['D:\MI\MI\BCIData\tsf_cv_data\A0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label');

end


        
