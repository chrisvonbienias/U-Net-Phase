classdef segRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = segRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Error for segmentation';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            
            difference1=Y-T;
            err_RMSE = rms(difference1(:));
            
            C = 10;
            difference2=((Y(:,:,1,:)+Y(:,:,2,:))/C-(T(:,:,1,:)+T(:,:,2,:))/C);
            err_sum=rms(difference2(:))/2;

            loss=(err_RMSE+err_sum)/2;

        end
        
        
    end
end