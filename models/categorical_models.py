categorical_vigilance_model = """
data { 
  int<lower=1> n_items;
  int<lower=1> n_annotators;
  int<lower=1> n_total_responses;
  int<lower=3> n_levels;
  vector[n_levels] priors;
  int<lower=1, upper=n_annotators> annotator_for_response[n_total_responses];
  int<lower=1, upper=n_items> item_for_response[n_total_responses];
  int<lower=1, upper=n_levels> responses[n_total_responses];
}
parameters {
  vector[n_levels] item_means[n_items];
  real<lower=0> items_std;
  vector[n_levels] annotator_offsets[n_annotators];
  real<lower=0, upper=1> vigilance[n_annotators];
  real<lower=0> offset_std;
}
model {
  // Priors
  items_std ~ normal(0, 1);
  for (i in 1:n_items) {
    for (k in 1:n_levels) { 
      item_means[i, k] ~ normal(priors[k], items_std);
    }
  }
  
  offset_std ~ normal(0, 1);  
  for (w in 1:n_workers) {
    for (k in 1:n_levels) {
      annotator_offsets[w] ~ normal(0, offset_std);
    }
  }
    
  for (r in 1:n_total_responses) {
    vector[n_levels] logits;
    for (k in 1:n_levels) {
        logits[k] = (vigilance[annotator_for_response[r]] * item_means[item_for_response[r], k] + (1-vigilance[annotator_for_response[r]]) * worker_offsets[annotator_for_response[r], k];
    ratings[r] ~ categorical_logit(logits);  
  }
}
"""
