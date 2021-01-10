data {
  int<lower=0> n;
  vector[n] temperature;
  int<lower=0,upper=1> damage[n];
}

parameters {
  real alpha;
  real beta;
}

model {
  alpha ~ normal(0, sqrt(1000));
  beta ~ normal(0, sqrt(1000));
  damage ~ bernoulli_logit(beta * temperature + alpha);
}

generated quantities {
  vector[n] yrep;
  for (i in 1:n){
    yrep[i] = bernoulli_logit_rng(beta * temperature[i] + alpha);
  }
}
