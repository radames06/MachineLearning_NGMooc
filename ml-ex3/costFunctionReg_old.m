function [J, grad] = costFunctionReg_old(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i=1:m
  htheta = sigmoid(X(i,:) * theta);
  J = J + (-y(i) * log(htheta) - (1-y(i)) * log(1 - htheta));
end;
J = J / m;
for j=2:size(theta)
  J = J + lambda / (2 * m) * (theta(j))^2;
end;

for j=1:size(theta(:))
  for i=1:m
    htheta = sigmoid(X(i,:) * theta);
    grad(j) = grad(j) + (htheta - y(i)) * X(i, j);
  end;
  grad(j) = grad(j) / m;
  if j>1
    grad(j) = grad(j) + lambda / m * theta(j);
  end;
end;




% =============================================================

end
