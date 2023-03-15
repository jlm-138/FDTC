function F = FDTC(data,minbucket,X,threshold)
%% Calculate the cluering result of FDTC
% F = FDTC(data,minbucket,X,threshold)
% Input: 
%    - data: data needs clustering, where rows represent instances, columns
%    represent features
%    - minbucket: minimum number of instances in the leaf node (1/10 of the
%    number of instances is recommended)
%    - X: number of clusters
%    - threshold: minimum acceptable Silhouette value during adjustment
%    (-0.2--0.1 is recommended)
% Output: 
%    - F: the first column is the membership and the second column is cluster label 
% Version 1.0 -- 2023/2/28
% Ref: Jiao Lianmeng*, Yang Haoyu, Liu Zhun-ga, Pan Quan. Interpretable fuzzy clustering using unsupervised fuzzy decision trees. Information Sciences, vol.611, pp.540-563, 2022.
[a,b]=size(data);
F=zeros(a,2);
At_Max = max(data(:,1:b));
At_Min = min(data(:,1:b));
for i = 1:a
    data(i,1:b) = (data(i,1:b) - At_Min)./(At_Max-At_Min);
end%归一�?
distance=zeros(a,a);%计算两两样本间距离，用于后续轮廓度量的计�?
for i=1:a-1%a为样本数
    for j=i+1:a
        for k=1:b%b是特征维�?
            distance(i,j)=distance(i,j)+(data(i,k)-data(j,k))^2;
        end
        distance(i,j)=distance(i,j)^0.5;
        distance(j,i)=distance(i,j);
    end
end
Data=[data ones(a,1) ones(a,1) zeros(a,1)];%数据集末第一列记录隶属度，第二列记录样本序号，方便定�?
for i=1:a
    Data(i,b+2)=i;
end
index=zeros(1,3);
index(1,1)=1;
index(1,2)=a;
daishu=0;
rule=zeros(1,b);%记录树的路径
node=1;%这一代的�?��侧节点编�?
Node=1;%这一代最右侧节点编号
rulebase=[];
Center=zeros(1,b);
Mem=[];
while daishu<ceil(X/2)
    daishu=daishu+1;
    t=Node;
    for k=node:Node
        index(k,3)=0;
        Flag=0;
        c=index(k,2)-index(k,1)+1;
        sizebin=ceil(c/6.2);%直方图个�?
        bin=1/sizebin;%直方图宽�?
        B=zeros(b,sizebin);%记录样本个数，防止某个区间样本太�?
        BM=zeros(b,sizebin);
        SCORE=zeros(1,b);
        for i=1:b
            if rule(k,i)~=0
                continue;%若该特征已经在之前的节点中划分过了，则不再进行划�?
            end
            for j=index(k,1):index(k,2)
                if Data(j,i)==0
                    B(i,1)=B(i,1)+1;
                    BM(i,1)=BM(i,1)+Data(j,b+1);
                    continue;
                elseif Data(j,i)==1
                    B(i,sizebin)=B(i,sizebin)+1;
                    BM(i,sizebin)=BM(i,sizebin)+Data(j,b+1);
                    continue;
                end
                B(i,ceil(Data(j,i)/bin))=B(i,ceil(Data(j,i)/bin))+1;
                BM(i,ceil(Data(j,i)/bin))=BM(i,ceil(Data(j,i)/bin))+Data(j,b+1);
            end
        end
%         bar(B(3,:));%消除注释，可以查看某�?��的直方图�?
        best=0;
        Gb=0;
        for fea=1:b
            if daishu~=1&&(isempty(find(find(rule(k,:))==fea))==0)%每个维度在一条路径上只划分一次哦
                continue;
            end
            E=zeros(1,sizebin);
            for i=2:sizebin-1
                flag=0;
                left=BM(fea,i);
                right=BM(fea,i);
                for j=1:i-1
                    if BM(fea,i-j)>=left
                        left=BM(fea,i-j);
                        flag=1;
                    else
                        break;
                    end
                end
                if flag==0||left==BM(fea,i)
                    E(1,i)=0;
                    continue;
                end
                flag=0;
                for j=1:sizebin-i
                    if BM(fea,i+j)>=right
                        right=BM(fea,i+j);
                        flag=1;
                    else
                        break;
                    end
                end
                if flag==0||right==BM(fea,i)
                    E(1,i)=0;
                    continue;
                end
                E(1,i)=(left-BM(fea,i)+right-BM(fea,i))/2/(1+BM(fea,i));%计算每个谷底的得�?
            end
            i=sizebin;
            while i>=2
                if E(1,i)==E(1,i-1)&&E(1,i)~=0
                    E(1,i)=0;%删除等高的相邻谷�?
                end
                i=i-1;
            end
            [J,I]=sort(E,'descend');
            [~,z]=size(find(J));
            I=I(:,1:z);
            I=sort(I);
%             z=floor(z/6);
            if isempty(I)==1
                continue;
            end
            M=zeros(1,z);
            Data(index(k,1):index(k,2),:)=sortrows(Data(index(k,1):index(k,2),:),fea);
            v=distance(Data(:,b+2),:);
            v=v(:,Data(:,b+2));
            Num=zeros(1,z+1);
            Index=zeros(z+1,2);
            for i=1:z
                if i==1
                    Num(1,i)=sum(B(fea,1:I(1,i)));
                    Index(i,1)=1;
                    Index(i,2)=Num(1,i);
                    continue;
                end
                Num(1,i)=sum(B(fea,I(1,i-1)+1:I(1,i)));
                Index(i,1)=Index(i-1,2)+1;
                Index(i,2)=Index(i-1,2)+Num(1,i);
            end
            Num(1,z+1)=sum(B(fea,I(1,i)+1:sizebin));
            Index(z+1,1)=Index(z,2)+1;
            Index(z+1,2)=Index(z,2)+Num(1,z+1);
            Index=Index+index(k,1)-1;
            C=find(Index(:,1)>Index(:,2));
            Index(C,:)=[];
            I(:,C)=[];
            [~,z]=size(I);
            num=zeros(1,2);
            num(1,2)=c;
            cen=zeros(z+1,b+1);
            for i=1:z+1
                cen(i,1:b)=sum(Data(Index(i,1):Index(i,2),1:b).*Data(Index(i,1):Index(i,2),b+1),1);
                cen(i,b+1)=sum(Data(Index(i,1):Index(i,2),b+1));
            end
            center=zeros(2,b+1);%计算各个簇的中心
            center(2,:)=sum(cen,1);
            value=zeros(a,z);
            cut=zeros(1,z);
            for i=1:z
                num(1,1)=num(1,1)+Num(1,i);
                num(1,2)=num(1,2)-Num(1,i);
                n=find(num<=minbucket);
                f=num;
                center(1,:)=center(1,:)+cen(i,:);
                center(2,:)=center(2,:)-cen(i,:);
                if isempty(n)==0
                    M(1,i)=-1;
                    continue;
                end
                G=I(1,i)*bin;
                cent=center;
                for m=1:2
                    cent(m,1:b)=cent(m,1:b)./cent(m,b+1);
                end
                cent=cent(:,1:b);
                d=2;
                in=zeros(2,2);
                in(1,1)=index(k,1);
                in(1,2)=in(1,1)+f(1,1)-1;
                in(2,1)=in(1,2)+1;
                in(2,2)=index(k,2);
                cut(1,i)=in(1,2);
                for m=1:Node
                    if index(m,3)==1
                        cent=[cent;Center(m,:)];
                        d=d+1;
                        y=index(m,2)-index(m,1)+1;
                        f=[f y];
                        in=[in;index(m,1:2)];
                    end
                end
                closest=zeros(1,d);
                if d==2
                    closest(1,1)=2;
                    closest(1,2)=1;
                else
                    dis=ones(d,d).*b;
                    for m=1:d-1
                        for p=m+1:d
                            clo=0;
                            for q=1:b
                                clo=clo+(cent(p,q)-cent(m,q))^2;
                            end
                            clo=sqrt(clo);
                            dis(m,p)=clo;
                            dis(p,m)=clo;
                        end
                    end
                    for m=1:d
                        [~,closest(1,m)]=min(dis(:,m));
                    end
                end
                for m=1:d
                    for n=in(m,1):in(m,2)
                        y=v(n,:);
                        Distance=sum(y(1,in(m,1):in(m,2)).'.*Data(in(m,1):in(m,2),b+1));
                        Distance=Distance/(sum(Data(in(m,1):in(m,2),b+1))-Data(n,b+1));
                        Dis=sum(y(1,in(closest(1,m),1):in(closest(1,m),2)).'.*Data(in(closest(1,m),1):in(closest(1,m),2),b+1));
                        Dis=Dis/sum(Data(in(closest(1,m),1):in(closest(1,m),2),b+1));
                        Data(n,b+3)=(Dis-Distance)/(max(Dis,Distance));
                    end
                end
                s=sum(Data(:,b+3))/a;
                value(index(k,1):index(k,2),i)=Data(index(k,1):index(k,2),b+3);
                M(1,i)=s;
            end
            y=max(M);
            if y<=0
                continue;
            end
            N=[];
            x=[];
            for i=1:z
                if M(1,i)>0.99*y
                    N=[N;I(1,i)];
                else
                    x=[x;i];
                end
            end
            value(:,x)=[];
            cut(:,x)=[];
            [e,~]=size(N);
            LR=zeros(e,2);
            for i=1:e
                LR(i,1)=B(fea,N(i,1))+B(fea,N(i,1)-1);
                LR(i,2)=B(fea,N(i,1)+1);
            end
            LR(LR==1)=2;
            LR(LR==0)=2;
            for i=1:e
                LR(LR(i,1)>cut(1,i)-index(k,1)+1-minbucket)=cut(1,i)-index(1,k)+1-minbucket;
                LR(LR(i,2)>index(k,2)-cut(1,i)-minbucket)=index(k,2)-cut(1,i)-minbucket;
            end
            for i=1:e
                Move=zeros(2,2);
                for j=1:2
                    if closest(1,j)~=3-j
                        continue;
                    end
                    if j==1
                        m=cut(1,i)-index(k,1)+1;
                        A=zeros(1,LR(i,1));
                        A(1,1)=value(cut(1,i),i);
                        for n=2:LR(i,1)
                            A(1,n)=A(1,n-1)+value(cut(1,i)-n+1,i);
                        end
                        if min(A)<0
                            [~,u]=min(A);
                            if u~=LR(i,1)
                                Move(1,j)=min(A);
                            else
                                while A(1,u)<0&&u<cut(1,i)-minbucket
                                    u=u+1;
                                    A(1,u)=A(1,u-1)+value(cut(1,i)-u+1,i);
                                    if m-u<=minbucket
                                        break;
                                    end
                                end
                                [Move(1,j),u]=min(A);
                            end
                        end
                    else
                        n=index(k,2)-cut(1,i)+1;
                        A=zeros(1,LR(i,2));
                        A(1,1)=value(cut(1,i)+1,i);
                        for p=2:LR(i,2)
                            A(1,p)=A(1,p-1)+value(p+cut(1,i),i);
                        end
                        if min(A)<0
                            [~,o]=min(A);
                            if o~=LR(i,2)
                                Move(1,j)=min(A);
                            else
                                while A(1,o)<0
                                    o=o+1;
                                    A(1,o)=A(1,o-1)+value(o+cut(1,i),i);
                                    if n-o<=minbucket
                                        break;
                                    end
                                end
                                [Move(1,j),o]=min(A);
                            end
                        end
                    end
                end
                if Move(1,1)<Move(1,2)
                    cut(1,i)=cut(1,i)-u;
                elseif Move(1,1)>Move(1,2)
                    cut(1,i)=cut(1,i)+o;
                end
            end
            cut=unique(cut);
            [~,e]=size(cut);
            cen=zeros(e+1,b+1);
            cen(1,:)=sum(Data(index(k,1):cut(1,1),1:b+1),1);
            for i=2:e
                cen(i,:)=sum(Data(cut(1,i-1)+1:cut(1,i),1:b+1),1);
            end
            cen(e+1,:)=sum(Data(cut(1,e)+1:index(k,2),1:b+1),1);
            for i=1:e
                x=nchoosek(cut,i);
                for j=1:(factorial(e)/factorial(i)/factorial(e-i))
                    G=x(j,:);
                    [~,d]=size(G);
                    Index=zeros(1,d);
                    for m=1:d
                        Index(1,m)=find(G(1,m)==cut);
                    end
                    num=zeros(1,d+1);
                    num(1,1)=G(1,1)-index(k,1)+1;
                    num(1,d+1)=index(k,2)-G(1,d)+1;
                    if d>=2
                        for m=2:d
                            num(1,m)=G(1,m)-G(1,m-1);
                        end
                    end
                    n=find(num<=minbucket);
                    if isempty(n)==0
                        continue;
                    end
                    Flag=Flag+1;
                    center=zeros(d+1,b+1);%计算各个簇的中心
                    center(1,:)=sum(cen(1:Index(1,1),:),1);
                    center(d+1,:)=sum(cen(Index(1,d)+1:e+1,:),1);
                    if d>=2
                        for m=2:d
                            center(m,:)=sum(cen(Index(1,m-1)+1:Index(1,m),:),1);
                        end
                    end
                    for m=1:d+1
                        center(m,1:b)=center(m,1:b)./center(m,b+1);
                    end
                    center=center(:,1:b);
                    dd=d;
                    in=zeros(d+1,2);
                    in(1,1)=index(k,1);
                    in(1,2)=G(1,1);
                    in(d+1,1)=G(1,d)+1;
                    in(d+1,2)=index(k,2);
                    if d>=2
                        for m=2:d
                            in(m,1)=G(1,m-1)+1;
                            in(m,2)=G(1,m);
                        end
                    end
                    for m=1:Node
                        if index(m,3)==1
                            center=[center;Center(m,:)];
                            in=[in;index(m,1:2)];
                            d=d+1;
                        end
                    end
                    closest=zeros(1,d+1);
                    dis=ones(d+1,d+1).*b;
                    for m=1:d
                        for p=m+1:d+1
                            clo=0;
                            for q=1:b
                                clo=clo+(center(p,q)-center(m,q))^2;
                            end
                            clo=sqrt(clo);
                            dis(m,p)=clo;
                            dis(p,m)=clo;
                        end
                    end
                    for m=1:d+1
                        [~,closest(1,m)]=min(dis(:,m));
                    end
                    for q=1:d+1
                        for n=in(q,1):in(q,2)
                            y=v(n,:);
                            Distance=sum(y(1,in(q,1):in(q,2)).'.*Data(in(q,1):in(q,2),b+1));
                            Distance=Distance/(sum(Data(in(q,1):in(q,2),b+1))-Data(n,b+1));
                            Dis=sum(y(1,in(closest(1,q),1):in(closest(1,q),2)).'.*Data(in(closest(1,q),1):in(closest(1,q),2),b+1));
                            Dis=Dis/sum(Data(in(closest(1,q),1):in(closest(1,q),2),b+1));
                            Data(n,b+3)=(Dis-Distance)/(max(Dis,Distance));
                        end
                    end
                    s=sum(Data(:,b+3))/a;
                    if s>SCORE(1,fea)
                        SCORE(1,fea)=s;
                    end
                    if Flag==1
                        best=SCORE(1,fea);
                        Gb=G;
                        Fea=fea;
                    else
                        if SCORE(1,fea)>best
                            best=SCORE(1,fea);
                            Gb=G;
                            Fea=fea;
                        end
                    end
                end
            end
        end
        if Gb==0
            index(k,3)=1;
            continue;
        end
        Data(index(k,1):index(k,2),:)=sortrows(Data(index(k,1):index(k,2),:),Fea);
        [~,d]=size(Gb);
        G=[];
        for i=1:d
            G(1,i)=(Data(Gb(1,i),Fea)+Data(Gb(1,i)+1,Fea))/2;
        end
        z=zeros(d+1,b+1);
        for i=1:d+1
            x=rule(k,:);
            x(1,Fea)=i;
            rule=[rule;x(1,:)];%产生子节点的形式
            Node=Node+1;
        end
        Membership=ones(c,d+1);
        for i=index(k,1):index(k,2)
            Membership(i,:)=membership(G,Data(i,Fea));%计算每个样本对于该模糊划分的隶属�?
        end
        mem=zeros(a,d+1);
        for i=1:a
            mem(i,:)=membership(G,data(i,Fea));
        end
        Mem=[Mem mem];
        y=ones(d+1,3);
        y(1,1)=index(k,1);
        y(1,2)=Gb(1,1);
        Data(index(k,1):Gb(1,1),b+1)=Data(index(k,1):Gb(1,1),b+1).*Membership(index(k,1):Gb(1,1),1);
        z(1,1:b)=sum(Data(index(k,1):Gb(1,1),1:b).*Data(index(k,1):Gb(1,1),b+1),1);
        z(1,b+1)=sum(Data(index(k,1):Gb(1,1),b+1));
        y(d+1,1)=Gb(1,d)+1;
        y(1+d,2)=index(k,2);
        Data(Gb(1,d)+1:index(k,2),b+1)=Data(Gb(1,d)+1:index(k,2),b+1).*Membership(Gb(1,d)+1:index(k,2),d+1);
        z(1+d,1:b)=sum(Data(Gb(1,d)+1:index(k,2),1:b).*Data(Gb(1,d)+1:index(k,2),b+1),1);
        z(1+d,b+1)=sum(Data(Gb(1,d)+1:index(k,2),b+1));
        if d>=2
            for i=2:d
                y(i,1)=Gb(1,i-1)+1;
                y(i,2)=Gb(1,i);
                Data(Gb(1,i-1)+1:Gb(1,i),b+1)=Data(Gb(1,i-1)+1:Gb(1,i),b+1).*Membership(Gb(1,i-1)+1:Gb(1,i),i);
                z(i,1:b)=sum(Data(Gb(1,i-1)+1:Gb(1,i),1:b).*Data(Gb(1,i-1)+1:Gb(1,i),b+1),1);
                z(i,b+1)=sum(Data(Gb(1,i-1)+1:Gb(1,i),b+1));
            end
        end
        index=[index;y];
        for i=1:d+1
            z(i,1:b)=z(i,1:b)./z(i,b+1);
        end
        z=z(:,1:b);
        Center=[Center;z];
    end
    node=t+1;
end
[e,~]=size(index);
Index=[];
for i=1:e
    if index(i,3)==1
        rulebase=[rulebase;rule(i,:)];
        Index=[Index;index(i,1:2)];
    end
end
[e,~]=size(Index);
Index=sortrows(Index,1);
Rule=cell(1,e);
for i=1:e
    Rule{1,i}=rulebase(i,:);
end
sminbefore=-1;
Score=zeros(1,e);
while 1
    f=zeros(1,e);
    for i=1:e
        f(1,i)=Index(i,2)-Index(i,1)+1;
    end
    center=zeros(e,b);%计算各个簇的中心
    for i=1:e
        for j=1:b
            center(i,j)=mean(Data(Index(i,1):Index(i,2),j));
        end
    end
    x=ones(e,e).*b;
    closest=zeros(e,1);%寻找每个簇离自己�?��的簇
    for i=1:e-1
        for j=i+1:e
            dis=0;
            for k=1:b
                dis=dis+(center(i,k)-center(j,k))^2;
            end
            dis=dis^0.5;
            x(i,j)=dis;
            x(j,i)=dis;
        end
    end
    for i=1:e
        [~,closest(i,1)]=min(x(i,:));
    end
    s=zeros(1,e);
    for i=1:e
        for j=Index(i,1):Index(i,2)
            Distance=0;%计算簇内样本间的距离
            for k=Index(i,1):Index(i,2)
                distance=0;
                if j==k
                    continue;
                end
                for n=1:b
                    distance=distance+(Data(j,n)-Data(k,n))^2;
                end
                distance=distance^0.5;
                Distance=Distance+distance;
            end
            Distance=Distance/(f(1,i)-1);
            Dis=0;%计算簇到�?��簇间得距�?
            for k=Index(closest(i,1),1):Index(closest(i,1),2)
                dis=0;
                for n=1:b
                    dis=dis+(Data(j,n)-Data(k,n))^2;
                end
                dis=dis^0.5;
                Dis=dis+Dis;
            end
            Dis=Dis/f(1,closest(i,1));
            Data(j,b+3)=(Dis-Distance)/(max(Dis,Distance));
        end
        s(1,i)=mean(Data(Index(i,1):Index(i,2),b+3));
    end
    flag=0;
    num=zeros(1,e);
    temp=[];
    for i=1:e
        for j=Index(i,1):Index(i,2)
            if Data(j,b+3)<threshold
                temp=[temp;Data(j,:)];
                Data(j,b+3)=-1;
                num(1,i)=num(1,i)+1;
                flag=1;
            end
        end
        Index(i,2)=Index(i,2)-sum(num(1,1:i));
        if i>1
            Index(i,1)=Index(i,1)-sum(num(1,1:i-1));
        end
    end
    Data(find(Data(:,b+3)==-1),:)=[];
    Num=zeros(e,2);
    for i=1:e
        Num(i,2)=sum(num(1,1:i));
        Num(i,1)=Num(i,2)-num(1,i)+1;
    end
    for i=1:e
        z=e+1-i;
        x=find(closest==z);
        [y,~]=size(x);
        for j=1:y
            if z~=e
                Data=[Data(1:Index(z,2),:);temp(Num(x(j,1),1):Num(x(j,1),2),:);Data(Index(z,2)+1:Index(e,2),:)];
                Index(z,2)=Index(z,2)+num(1,x(j,1));
                Index(z+1:e,:)=Index(z+1:e,:)+num(1,x(j,1));
            else
                Data=[Data(1:Index(z,2),:);temp(Num(x(j,1),1):Num(x(j,1),2),:)];
                Index(z,2)=Index(z,2)+num(1,x(j,1));
            end
        end
    end
    if flag==1
        continue;
    end
    [smin,p]=min(s);
    score=zeros(1,e);
    for i=1:e
        if (s(1,i)-smin)<0.05&&i~=p&&((closest(i,1)==closest(p,1)&&closest(closest(p,1),1)==i)||closest(p,1)==i)
            p=i;
        end
        score(1,i)=s(1,i)*f(1,i)/a;
    end
    Score(1,e)=sum(score);
    if e>=X
        if X==2&&e==2
            break;
        end
        Databefore=Data;
        Indexbefore=Index;
        t=Data(Index(p,1):Index(p,2),:);
        Data(Index(p,1):Index(p,2),:)=[];
        if p~=e
            Index(p+1:e,:)=Index(p+1:e,:)-f(1,p);
        end
        Index(p,:)=zeros(1,2);
        if closest(p,1)==e
            Data=[Data;t];
            Index(closest(p,1),2)=Index(closest(p,1),2)+f(1,p);
        else
            Data=[Data(1:Index(closest(p,1),2),:);t;Data(Index(closest(p,1),2)+1:a-f(1,p),:)];
            Index(closest(p,1),2)=Index(closest(p,1),2)+f(1,p);
            Index(closest(p,1)+1:e,:)=Index(closest(p,1)+1:e,:)+f(1,p);
        end
        Index(find(Index(:,1)==f(1,p)),:)=[];
        Index(find(Index(:,1)==0),:)=[];
        Rulebefore=Rule;
        Rule{1,closest(p,1)}=[Rule{1,closest(p,1)};Rule{1,p}];
        Rule(:,p)=[];
        fbefore=f;
        f(1,closest(p,1))=f(1,closest(p,1))+f(1,p);
        f(:,p)=[];
        e=e-1;
        sminbefore=smin;
    else
        if sminbefore==-1
            break;
        end
        Data=Databefore;
        Index=Indexbefore;
        Rule=Rulebefore;
        f=fbefore;
        e=e+1;
        break;
    end
end
for i=1:e
    for j=1:f(1,i)
        if i==1
            F(Data(j,b+2),2)=i;
            F(Data(j,b+2),1)=Data(j,b+1);
        else
            F(Data(j+sum(f(1,1:i-1)),b+2),2)=i;
            F(Data(j+sum(f(1,1:i-1)),b+2),1)=Data(j+sum(f(1,1:i-1)),b+1);
        end
    end
end
end