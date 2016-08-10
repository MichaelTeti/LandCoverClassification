clc;
close all;
clear all;


ps=8;
im=im2double(imread('earth.jpg'));
im=im(1:3:end, 1:3:end, :);
x=im2col(im, [ps ps], 'sliding');

% load('patches.mat');
% x=patches;
k=200;
m=size(x, 2);

a=randn(k, m);
r=randperm(m);
old_dict=x(:, r(1:k));

for iters=1:100
    cd('/home/mpcr/Desktop/Indian_pines');
    sprintf('Iteration: %d', iters)
    t=.01;
    h=.0001;
    d=h/t;
    lambda=.01;
    u=zeros(size(a));
	    
    % locally competitive neural network
    for i=1:100
        a=(u-sign(u).*(lambda)).*(abs(u)>(lambda));
        u=u+d*(old_dict'*(x-old_dict*a)-u-a);
    end     
    
    % update only group of blocks with most change
    % during the iteration
    bs=9;
    D=x*pinv(a);
    [r g]=find(D==0);
    D(r, g)=.0001;
    D=D./repmat(sqrt(sum(D.^2)),[size(D, 1), 1]);
    diff=abs(D-old_dict);
    blocks=im2col(diff, [ps*ps bs], 'distinct');
    blocks=sum(blocks);
    b=find(max(blocks));
    old_dict(:, b*bs-(bs-1):b*bs)=D(:, b*bs-(bs-1):b*bs);
    cd('/home/mpcr/Desktop/Indian_pines/spams-matlab');
    start_spams;
    displayPatches(old_dict); colormap('gray');
end 

save('dictionary.mat', 'old_dict');
    
