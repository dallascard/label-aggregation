basic_poisson_model = """
data { 
  int<lower=1> n_items;
  int<lower=1> n_annotators;
  int<lower=1> n_total_responses;
  int<lower=1, upper=n_annotators> annotator_for_response[n_total_responses];
  int<lower=1, upper=n_items> item_for_response[n_total_responses];
  int<lower=0> responses[n_total_responses];
}
parameters {
  vector[n_items] item_means;
  real<lower=0> item_std;
  vector[n_annotators] annotator_offsets;
  real<lower=0> offset_std;
}
model {
  // Priors  
  //item_std ~ normal(0, 1);
  //item_means ~ normal(0, item_std);
  
  offset_std ~ normal(0, 1);    
  for (a in 1:n_annotators) {
    annotator_offsets[a] ~ normal(0, offset_std);

  }  
  for (r in 1:n_total_responses) {
    real mu;
    mu = item_means[item_for_response[r]] + annotator_offsets[annotator_for_response[r]];
    responses[r] ~ poisson_log(mu);  
  }
}
"""

basic_nb_model = """
data { 
  int<lower=1> n_items;
  int<lower=1> n_annotators;
  int<lower=1> n_total_responses;
  int<lower=1, upper=n_annotators> annotator_for_response[n_total_responses];
  int<lower=1, upper=n_items> item_for_response[n_total_responses];
  int<lower=0> responses[n_total_responses];
}
parameters {
  vector[n_items] item_means;
  real<lower=0> item_std;
  vector[n_annotators] annotator_offsets;
  real<lower=0> offset_std;
  real<lower=0> phi;
}
model {
  // Priors  
  //item_std ~ normal(0, 5);
  //item_means ~ normal(0, item_std);

  phi ~ normal(0, 5);
  
  offset_std ~ normal(0, 1);    
  for (a in 1:n_annotators) {
    annotator_offsets[a] ~ normal(0, offset_std);

  }  
  for (r in 1:n_total_responses) {
    real mu;
    mu = item_means[item_for_response[r]] + annotator_offsets[annotator_for_response[r]];
    responses[r] ~ neg_binomial_2_log(mu, phi);  
  }
}
"""