function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Using X, y, lambda, Theta1, Theta2
for (k = 1:m) % compute cost for each training example
    a1 = [1, X(k, :)]'; % 401 x 1 vector; input layer
    
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; % 26 x 1 vector; hidden layer
    
    z3 = Theta2 * a2;
    hx_k = sigmoid(z3); % output: hypothesis column vector
    
    y_k = zeros(num_labels, 1);
    y_k(1:num_labels == y(k)) = 1;
    
    j1 = -1 * y_k' * log(hx_k);
    j2 = -1 * (1 - y_k)' * log(1 - hx_k);
    
    % Non-regularized cost
    J = J + (1/m) * (j1 + j2);
    
    % Gradient Descent
    d3 = hx_k - y_k;
    d2 = Theta2' * d3;
    d2 = d2(2:end) .* sigmoidGradient(z2); % 25 x 1 vector
    Theta2_grad = Theta2_grad + d3 * a2';
    Theta1_grad = Theta1_grad + d2 * a1';
end 

% Now we will add the regularization terms
reg1 = sum(sum(Theta1(:, 2:end) .^ 2));
reg2 = sum(sum(Theta2(:, 2:end) .^ 2));
J = J + (lambda / (2 * m)) * (reg1 + reg2);

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda) .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda) .* Theta2(:, 2:end);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% multiply by 1/m
grad = (1 / m) .* grad;

end
