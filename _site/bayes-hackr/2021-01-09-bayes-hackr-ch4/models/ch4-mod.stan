data {
  int<lower=0> n;
  int upvotes[n];
  int total_votes[n];
}

parameters {
  real<lower=0,upper=1> upvote_ratio;
}

model {
  upvote_ratio ~ uniform(0, 1);
  upvotes ~ binomial(total_votes, upvote_ratio);
}
