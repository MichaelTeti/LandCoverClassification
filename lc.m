clc;
close all;
clear all;

ps=8;
%data=load('Indian_pines.mat');
%data=struct2cell(data);
%data=cell2mat(data);
%x=im2col(data, [ps ps], 'sliding');
load('patches.mat');
x=patches;
k=200;
m=size(x, 2);

alpha=randn(k, m);
r=randperm(m);
old_dict=x(:, r(1:k));

for iters=1:10
    t=.01;
    h=.0001;
    d=h/t;
    lambda=.01;
    u=zeros(size(alpha));
    
    for i=1:300
        alpha=(u-sign(u).*(lambda)).*(abs(u)>(lambda));
        u=u+d*(old_dict'*(x-old_dict*alpha)-u-alpha);
    end 
    
    D=x*pinv(alpha);
    D=D./repmat(sqrt(sum(D.^2)),[size(D, 1), 1]);
    diff=abs(D-old_dict);
    blocks=im2col(diff, [ps 4], 'distinct');
    blocks=sum(blocks);
    b=find(max(blocks));
    old_dict(:, w*2+1:w*4)=D(:, w*2+1:w*4);
    start_spams;
    displayPatches(old_dict); colormap('gray');
    pause;
end 
    
