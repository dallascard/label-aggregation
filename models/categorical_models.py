basic_categorical_model = """
data { 
  int<lower=1> n_items;
  int<lower=1> n_annotators;
  int<lower=1> n_total_responses;
  int<lower=3> n_levels;
  vector[n_levels] priors;
  array[n_total_responses] int<lower=1, upper=n_annotators> annotator_for_response;
  arrary[n_total_responses] int<lower=1, upper=n_items> item_for_response;
  array[n_total_responses] int<lower=1, upper=n_levels> responses;
}
parameters {
  vector[n_levels] item_means[n_items];
  real<lower=0> item_std;
  vector[n_levels] annotator_offsets[n_annotators];
  real<lower=0> offset_std;
}
model {
  // Priors
  item_std ~ normal(0, 1);
  for (i in 1:n_items) {
    for (k in 1:n_levels) { 
      item_means[i, k] ~ normal(priors[k], item_std);
    }
  }
  
  offset_std ~ normal(0, 1);  
  for (a in 1:n_annotators) {
    for (k in 1:n_levels) {
      annotator_offsets[a, k] ~ normal(0, offset_std);
    }
  }
    
  for (r in 1:n_total_responses) {
    vector[n_levels] logits;
    for (k in 1:n_levels) {
      logits[k] = item_means[item_for_response[r], k] + annotator_offsets[annotator_for_response[r], k];
    }
    responses[r] ~ categorical_logit(logits);  
  }
}
"""


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
  real<lower=0> item_std;
  vector[n_levels] annotator_offsets[n_annotators];
  real<lower=0, upper=1> vigilance[n_annotators];
  real<lower=0> offset_std;
}
model {
  // Priors
  item_std ~ normal(0, 1); 
  for (i in 1:n_items) {
    for (k in 1:n_levels) { 
      item_means[i, k] ~ normal(priors[k], item_std);
    }
  }
  
  offset_std ~ normal(0, 1);  
  for (a in 1:n_annotators) {
    for (k in 1:n_levels) {
      annotator_offsets[a, k] ~ normal(0, offset_std);
    }
  }
    
  for (r in 1:n_total_responses) {
    vector[n_levels] logits;
    for (k in 1:n_levels) {
      logits[k] = vigilance[annotator_for_response[r]] * item_means[item_for_response[r], k] + (1-vigilance[annotator_for_response[r]]) * annotator_offsets[annotator_for_response[r], k];
    }
    responses[r] ~ categorical_logit(logits);  
  }
}
"""
