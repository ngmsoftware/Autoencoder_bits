clear('all');

HEX = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' ];

I = imread('~/Documents/Digits.png');

I = double(I(:,:,1))/255;

I = I(740:end, 513:end);

I1 = I(1:570,1:570);
I2 = I(1:570,570:(570+569));

amountDigits = containers.Map({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'},{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});

n = 7; 
for i=1:10
    for j=1:10
        subI = I1(1+(i-1)*57:i*57, 1+(j-1)*57:j*57);
        subI(:,1) = 1;
        subI(:,end) = 1;
        subI(1,:) = 1;
        subI(end,:) = 1;
        
        subI = imresize(subI, [28, 28]);
        
        imagesc(subI);
        digit = HEX(n+1);
        
        amountDigits(digit) = amountDigits(digit)+1;
        
        name = [digit '-' num2str(amountDigits(digit)) '.png'];

        title(name);

        imwrite(subI, name);
        
        pause(0.01);
        
        
        n = mod(n+1,16);
    end
end


n = 1; 
for i=1:10
    for j=1:10
        subI = I2(1+(i-1)*57:i*57, 1+(j-1)*57:j*57);
        subI(:,[1, 2]) = 1;
        subI(:,end) = 1;
        subI([1, 2],:) = 1;
        subI(end,:) = 1;
        
        subI = imresize(subI, [28, 28]);
        
        imagesc(subI);
        digit = HEX(n+1);

        amountDigits(digit) = amountDigits(digit)+1;

        name = [digit '-' num2str(amountDigits(digit)) '.png'];
        
        title(name);

        imwrite(subI, name);
        
        pause(0.01);
        
        
        n = mod(n+1,16);
    end
end


