valeurs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error_min = 10^16;
for c_local = valeurs
  for sigma_local = valeurs
    fprintf('c = %f   ;    sigma = %f\n',c_local, sigma_local);
    model = svmTrain(X, y, c_local, @(x1, x2) gaussianKernel(x1, x2, sigma_local)); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if error_min > error
      error_min = error;
      C_min = c_local;
      sigma_min = sigma_local;
      fprintf('************************  Nouveau min : error = %f\n', error);
    end;
  end;
end;
fprintf('RESULTAT : error = %f   ;   c_min = %f   ;    sigma_min = %f\n',error_min,C_min, sigma_min);
