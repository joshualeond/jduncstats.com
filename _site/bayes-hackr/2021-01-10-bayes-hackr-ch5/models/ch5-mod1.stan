parameters {
  real true_price;
  real prize_1;
  real prize_2;
}

transformed parameters {
  real price_estimate;
  price_estimate = prize_1 + prize_2;
}

model {
  // priors
  true_price ~ normal(35000, 7500);
  prize_1 ~ normal(3000, 500);
  prize_2 ~ normal(12000, 3000);
  // updated price using priors of individual prizes
  true_price ~ normal(price_estimate, 3000);
}
