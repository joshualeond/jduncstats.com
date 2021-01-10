functions {
  int sign(real x) {
    return x < 0 ? -1 : 1;
  }
  
  real stock_loss(real true_return, real yhat) {
    real alpha = 100;
    if (true_return * yhat < 0)
      return(alpha * yhat^2 - sign(true_return) * yhat + fabs(true_return));
    else
      return(fabs(true_return - yhat));
  }
}

data {
  int<lower=0> n;
  vector[n] X;
  vector[n] Y;
  vector[50] trading_signals;
}

parameters {
  real beta;
  real alpha;
  real<lower=0> std;
}

model {
  alpha ~ normal(0, 100);
  beta ~ normal(0, 100);
  std ~ uniform(0, 100);
  Y ~ normal(alpha + beta * X, std);
}

generated quantities {
   vector[50] outcomes;
   vector[50] util;
   
   for (i in 1:50){
     outcomes[i] = normal_rng(alpha + beta * trading_signals[i], std);
     util[i] = stock_loss(trading_signals[i], outcomes[i]);
   }
}
