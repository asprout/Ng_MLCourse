function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
testVals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
testErrors = zeros(length(testVals) ^ 2, 3);

for i_testC = 1:length(testVals)
    for i_testSig = 1:length(testVals)
        rowNum = (i_testC - 1) * length(testVals) + i_testSig;
        testC = testVals(i_testC);
        testSig = testVals(i_testSig);
        model = svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSig));
        preds = svmPredict(model, Xval);
        testErrors(rowNum, 1) = testC;
        testErrors(rowNum, 2) = testSig;
        testErrors(rowNum, 3) = mean(double(preds ~= yval));
    end
end

[mini, mind] = min(testErrors(:, 3));
% =========================================================================
C = testErrors(mind, 1);
sigma = testErrors(mind, 2);
end
