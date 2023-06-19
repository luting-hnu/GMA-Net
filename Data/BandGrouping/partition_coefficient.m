% for i=1:103
% im(:,:,i)=(paviaU(:,:,i)-min(min(paviaU(:,:,i))))./(max(max(paviaU(:,:,i)))-min(min(paviaU(:,:,i))));
% end
% 
% for i=1:103
% for j=1:103
% b = corrcoef(im(:,:,i), im(:,:,j));
% corr_matrix(i)=b(1,2);
% corr_matrix2(i,j)=b(1,2);
% end
% end

% HSI = paviaU;
% m = 610;
% n = 340;
% c = 103;
clc;clear all
%=====================================================
% load 'PaviaU.mat';
% HSI = paviaU;

% load 'Houston.mat';
% HSI = img;

% load 'Indian_pines_corrected.mat';
% HSI = indian_pines_corrected;

load 'Salinas_corrected.mat';
HSI = salinas_corrected;

% load 'Botswana.mat';
% HSI = Botswana;

% load 'KSC.mat';
% HSI = KSC;
%=====================================================

HSI_size = size(HSI);
m = HSI_size(1);
n = HSI_size(2);
c = HSI_size(3);

% 逐波段展平
im2 = double(zeros(c,m*n));
for i=1:c
    im2(i,:) = reshape(HSI(:,:,i),1,m*n);
end

% 逐波段归一化
for i=1:c
    im2(i,:)=(im2(i,:)-min(im2(i,:)))./(max(im2(i,:))-min(im2(i,:)));
end

% 计算邻近波段的一维相关性向量
im2_corr1 = ones(c-1,1);
for i=1:c-1
   b = corrcoef(im2(i,:), im2(i+1,:));
   im2_corr1(i)=b(1,2);
end
% figure(1)
% plot(1:c-1,im2_corr1)

im2_corr1_mean = mean(im2_corr1);
% im2_corr0=[im2_corr1(1);im2_corr1];
% im2_corr1=[im2_corr1;im2_corr1(end)];
% im2_corr_diff=im2_corr1-im2_corr0;

localMinPoints = ones();  % 初始局部极小值点
%=======================MinDiff=================================
MinDiff =0.0025; % 0.005; %0.01 0.002; %平均相关性差异必须大于MinDiff，才算分割点
%=======================MinDiff=================================
critical_value = im2_corr1_mean*0.05; %0.05; %相关性差异小于critical_value，直接算为分割点分割点
for i=5:c-6 %c-2
    %找出所有的局部极小值点，即初始分割点
   if(im2_corr1(i)<im2_corr1(i-1))
       if(im2_corr1(i)<im2_corr1(i+1))
           left_average = (im2_corr1(i-4)+im2_corr1(i-3)+im2_corr1(i-2)+im2_corr1(i-1))/4;
           right_average = (im2_corr1(i+1)+im2_corr1(i+2)+im2_corr1(i+3)+im2_corr1(i+4))/4;
           if left_average-im2_corr1(i)>MinDiff && right_average-im2_corr1(i)>MinDiff % 相关性差异必须大于MinDiff，才算分割点
               localMinPoints = [localMinPoints i];
%            elseif im2_corr1(i)<=critical_value
%                localMinPoints = [localMinPoints i];
           end
       end
   end
end
localMinPoints = [localMinPoints c]; %加上末尾
% localMinPoints
localMinPoints
flag = 0; % 终止条件
%=======================setCorrValue=================================
setCorrValue = 0.995;  %相关性大于setCorrValue，则考虑合并分组
localMinPoints_size = size(localMinPoints); %[1,22] 一共22个点   1...103
numGroups = localMinPoints_size(2)-1; % 波段被分隔为多少组 21
numGroups
% 组内求平均，求组间相关性合并
%=======================Min_interval=================================
Min_interval = uint8(c*0.1); %分组最小间隔，如果间隔小于Min_interval，则考虑合并分组

while(flag==0)
    %计算所有极值点分隔后各个段的波段平均值,构造新图像
    newIm = ones(numGroups,m*n);  % 最终得到numGroups=21个新图像
    for i=1:numGroups
        length = localMinPoints(i+1)-localMinPoints(i);
        if i~=numGroups
            if length>10 %10  % 当分段内波段数太多，只取其中间的50%的波段计算平均值
                newIm(i,:) = sum(im2(localMinPoints(i)+(length*0.25):localMinPoints(i)+(length*0.75)-1,:))./(length*0.5); %取其中间的50%的波段计算平均值
                
            else
                newIm(i,:) = sum(im2(localMinPoints(i):localMinPoints(i+1)-1,:))./(localMinPoints(i+1)-localMinPoints(i));
            end
        else %最后一组情况特殊
            if length>10 %10  % 当分段内波段数太多，只取其中间的50%的波段计算平均值
                newIm(i,:) = sum(im2(localMinPoints(i)+(length*0.25):localMinPoints(i)+(length*0.75)-1,:))./(length*0.5); %取其中间的50%的波段计算平均值
            else
                newIm(i,:) = sum(im2(localMinPoints(i):localMinPoints(i+1),:))./(localMinPoints(i+1)-localMinPoints(i)+1);
            end
        end
    end
    %计算波段平均值之间的相关性系数
    newIm_corr = zeros(numGroups-1,1);
    for i=1:(numGroups-1)
        b = corrcoef(newIm(i,:), newIm(i+1,:));
        newIm_corr(i)=b(1,2);    % 最终得到numGroups-1=20个相关性系数
    end

    for i=1:(numGroups-1)
            %如果相关性系数大于setCorrValue，就删除该分割点
        if newIm_corr(i)>=setCorrValue && im2_corr1(i)>critical_value  %如果相关性大于临界值
            localMinPoints(i+1)=[];        %则删除该分割点
            numGroups = numGroups-1;
            break;
        end
             %如果间隔太小，就删除该分割点
        if localMinPoints(1,i+1)-localMinPoints(1,i)<=Min_interval && localMinPoints(1,i+2)-localMinPoints(1,i+1)<=Min_interval && im2_corr1(i)>critical_value
            if (newIm_corr(i) >= newIm_corr(i+1)) %改段与前一段更相关
                localMinPoints(i+1)=[];        %则删除该分割点
                numGroups = numGroups-1;
            elseif (newIm_corr(i+1) >= newIm_corr(i) && i~=numGroups-1  && im2_corr1(i)>critical_value)
                localMinPoints(i+2)=[];        %则删除该分割点
                numGroups = numGroups-1;
            end
            break;
        end
        
        if (localMinPoints(1,i+1)-localMinPoints(1,i)<Min_interval && im2_corr1(i)>critical_value)  % 如果该分割点前面的组很小
            localMinPoints(i+1)=[];        %则删除该分割点
            numGroups = numGroups-1;
            break;
        end
        
        if (localMinPoints(1,i+2)-localMinPoints(1,i+1)<Min_interval && i~=numGroups-1 && im2_corr1(i)>critical_value)  % 如果该分割点后面的组很小
            localMinPoints(i+2)=[];        %则删除后面的分割点
            numGroups = numGroups-1;
            break;
        end
        
        if i ==(numGroups-1)
            flag=1; %如果已经遍历了所有的i，此时i是最后一个，=(numGroups-1)，说明所有相关性系数大于setCorrValue的点都被删除，就将flag设为1
        end
        if numGroups == 1 || max(size(localMinPoints))==2
            flag=1; %如果分组到最后只剩下一个组
        end
    end
end
numGroups
localMinPoints

figure(1)
x = 1:c;       %[1×204]
y = im2_corr1';  %[1×203]
y = [y y(1,c-1)];
plot(x,y,'b','LineWidth',1.5);
% grid on
xlabel('Band','FontSize', 12) 
ylabel('光谱相关性','FontSize', 16)
localMinPoints = reshape(localMinPoints,1,[]);  %[1×6]
% xp = find(x==localMinPoints);
% yp = find(y==localMinPoints);

% text(localMinPoints,y(localMinPoints),num2str([localMinPoints;y(localMinPoints)].','(%.0f,%.2f)'),'color','r')
text(localMinPoints,y(localMinPoints),'x','color','r')

% text(x(localMinPoints),y(localMinPoints),[num2str(x(localMinPoints)),num2str(y(localMinPoints))]','(%.2f,%.2f)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
% % 间隔太小合并
% numGroups = numGroups; %合并前，此时共需要分numGroups个组     7
% % localMinPoints [1,58,73,86,89,97,100,103;] 8
% % newIm_corr
% % [0.804622488054651;0.511265078320287;0.873453345585097;0.998455682480242;0.997545249208789;0.997187627420801;0.998016938918438;]
% % 7
% 
% Min_interval = 20;
% delet_points = [];
% for i=2:numGroups  % 2:7
%     % 如果该分割点前后两个组都很小，通过比较相关性系数，将该组合并到相关性大的一边
%     if localMinPoints(1,i)-localMinPoints(1,i-1)<=Min_interval && localMinPoints(1,i+1)-localMinPoints(1,i)<=Min_interval
%         if (newIm_corr(i-1) >= newIm_corr(i)) %改段与前一段更相关
% %             localMinPoints(i)=[];        %则删除该分割点
% %             numGroups = numGroups-1;
%             delet_points = [delet_points i];
% %             i=i-1;
% %             break;
%         elseif (newIm_corr(i-1)<newIm_corr(i))
% %             localMinPoints(i+1)=[];        %则删除后面的分割点
% %             numGroups = numGroups-1;
%             delet_points = [delet_points i+1];
% %             i=i-1;
% %             break;
%         end
%         
%     elseif (localMinPoints(1,i)-localMinPoints(1,i-1)<Min_interval)  % 如果该分割点前面的组很小
% %         localMinPoints(i)=[];        %则删除该分割点
% %         numGroups = numGroups-1;
%         delet_points = [delet_points i];
% %         i=i-1;
% %         break;
%         
%     elseif (localMinPoints(1,i+1)-localMinPoints(1,i)<Min_interval)  % 如果该分割点后面的组很小
% %         localMinPoints(i+1)=[];        %则删除后面的分割点
% %         numGroups = numGroups-1;
%         delet_points = [delet_points i+1];
% %         i=i-1;
% %         break;
%     end
% end
% 
% % 计算全部波段的二维相关性矩阵
% im2_corr2 = ones(103,103);
% for i=1:102
%    b = corrcoef(im2(i,:), im2(i+1,:));
%    im2_corr1(i)=b(1,2);
% end
% 
% 
% for i=1:103
%     for j=1:103      
%         a =corrcoef(im2(i,:), im2(j,:)); 
%         im2_corr2(i,j) =a(1,2);
%     end
% end
% 
% figure(1)
% plot(1:102,im2_corr1)
% figure(2)
% imshow(uint8(im2_corr2*255))
% figure(3)
% imagesc(im2_corr2)



