data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  real y[N];               // observations
}

parameters {
  simplex[K] theta;          // mixing proportions
  ordered[K] mu;             // locations of mixture components
  vector<lower=0>[K] sigma;  // scales of mixture components
}

model {
  vector[K] log_theta = log(theta);  // cache log calculation
  sigma ~ uniform(0, 100);
  mu[1] ~ normal(120, 10);
  mu[2] ~ normal(190, 10);
  
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K)
      lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    target += log_sum_exp(lps);
  }
}

generated quantities {
  vector[N] yrep;
  for (i in 1:N){
    vector[K] log_theta = log(theta);
    yrep[i] = (normal_lpdf(y[i] | mu[1], sigma[1]) + log_theta[1]) >
    (normal_lpdf(y[i] | mu[2], sigma[2]) + log_theta[2]);
  }
}
