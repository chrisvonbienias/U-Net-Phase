clc
imi = 100;

Y = predict(network,inputTestImages(:,:,:,imi)+3);
X = predict(unet,inputTestImages(:,:,:,imi)+3);

input = inputTestImages(:,:,1,imi)+3;
output_1 = outputTestImages(:,:,1,imi);
output_2 = outputTestImages(:,:,2,imi);
Y_output_1 = Y(:,:,1);
Y_output_2 = Y(:,:,1);
X_output_1 = X(:,:,1);
X_output_2 = X(:,:,1);

disp('RMSE Y1')
RMSE2(output_1, Y_output_1)

disp('RMSE Y2')
RMSE2(output_2, Y_output_2)

disp('RMSE Y')
RMSE2(output_1 + output_2, Y_output_1 + Y_output_2)

disp('RMSE X1')
RMSE2(output_1, X_output_1)

disp('RMSE X2')
RMSE2(output_2, X_output_2)

disp('RMSE X')
RMSE2(output_1 + output_2, X_output_1 + X_output_2)

function [mse, rmse] = RMSE2(signal1, signal2)
    originalRowSize = size(signal1,1);
    originalColSize = size(signal1,2);
    signal1 = signal1(:);
    signal2 = signal2(:);
    mse = sum((signal1 - signal2).^2)./(originalRowSize*originalColSize);
    rmse = sqrt(mse);
end
