functions {
real f_distance(vector gxy_pos, vector halo_pos, real c) {
  return fmax(distance(gxy_pos, halo_pos), c);
}

vector tangential_distance(vector glxy_position, vector halo_position) {
  vector[2] delta = glxy_position - halo_position;
  real t = (2 * atan(delta[2] / delta[1]));
  return to_vector({-cos(t), -sin(t)});
}
}

data {
  int<lower=0> n;
  matrix[2, n] cart_pos; // x,y coordinates of galaxy position
  matrix[2, n] ellip_pos; // a measure of ellipticity
}

parameters {
  real<lower=40.0,upper=180.0> exp_mass_large;
  vector<lower=0,upper=4200.0>[2] halo_position;
}

transformed parameters {
  real mass_large = log(exp_mass_large); // one large halo
}

model {
  vector[2] mu; 
  
  for (i in 1:n) {
    mu = mass_large / f_distance(cart_pos[:, i], halo_position, 240.0) * 
      tangential_distance(cart_pos[:, i], halo_position); 
    ellip_pos[, i] ~ normal(mu, 0.05); // x-y coordinates
  }
}
