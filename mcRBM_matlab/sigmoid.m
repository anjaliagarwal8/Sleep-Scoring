%% Function to find sigmoid of the input data.
% sigmoid(x) = 1/(1+e(-x))

function s = sigmoid(x)

    s = 1.0 ./ (1.0 + exp(-x));
    
end