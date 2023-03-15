function [ membership ] = membership( I,x )
%将硬划分模糊化，并得到点x的隶属度
[~,b]=size(I);
membership=zeros(1,b+1);
M=zeros(1,b+1);
for i=1:b-1
    M(1,i+1)=(I(1,i)+I(1,i+1))/2;
end
M(1,b+1)=1;
for i=1:b
    if x<=M(1,i+1)
        if (I(1,i)-M(1,i))<(M(1,i+1)-I(1,i))
            if (x>=M(1,i)&&x<(2*I(1,i)-M(1,i)))
                membership(1,i)=0.5-0.5*(x-I(1,i))/(I(1,i)-M(1,i));
                membership(1,i+1)=1-membership(1,i);
            else
                membership(1,i)=0;
                membership(1,i+1)=1;
            end
        else 
            if (x<=M(1,i+1)&&x>(2*I(1,i)-M(1,i+1)))
                membership(1,i+1)=0.5+0.5*(x-I(1,i))/(M(1,i+1)-I(1,i));
                membership(1,i)=1-membership(1,i+1);
            else
                membership(1,i)=1;
                membership(1,i+1)=0;
            end
        end
        break;
    end
end
end