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
   for (i in 1:50)
     outcomes[i] = normal_rng(alpha + beta * trading_signals[i], std);
}