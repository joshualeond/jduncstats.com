data {
  int<lower=1> n;
  int<lower=0> count[n];
  real alpha;
}

transformed data {
  real log_unif;
  log_unif = -log(n);
}

parameters {
  real<lower=0> lambda_1;
  real<lower=0> lambda_2;
}

transformed parameters {
  vector[n] lp;
  lp = rep_vector(log_unif, n);
  for (tau in 1:n)
    for (i in 1:n)
      lp[tau] += poisson_lpmf(count[i] | i < tau ? lambda_1 : lambda_2);
}

model {
  lambda_1 ~ exponential(alpha);
  lambda_2 ~ exponential(alpha);
  target += log_sum_exp(lp);
}

generated quantities {
  int<lower=1,upper=n> tau;
  vector<lower=0>[n] expected;
  // posterior of discrete change point
  tau = categorical_logit_rng(lp);
  // predictions for each day
  for (day in 1:n)
    expected[day] = day < tau ? lambda_1 : lambda_2;
}
