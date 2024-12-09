# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:42:42 2024

@author: steve
"""

import numpy as np
import pandas as pd
from scipy import stats
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from enum import Enum, auto

class DistributionType(Enum):
    GAMMA = auto()
    LOGNORMAL = auto()
    WEIBULL = auto()
    BETA = auto()

@dataclass
class DistributionFit:
    type: DistributionType
    params: tuple
    ks_statistic: float
    aic: float  

def clean_story_points(df):
    # Convert to numeric
    df['Story Points'] = pd.to_numeric(df['Story Points'], errors='coerce')
    
    # Remove rows with NaN story points
    df_clean = df.dropna(subset=['Story Points'])
    
    # Ensure positive values (add small constant to zeros)
    df_clean['Story Points'] = df_clean['Story Points'].clip(lower=0.1)
    
    print(f"Original dataset size: {len(df)}")
    print(f"Clean dataset size: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} rows with invalid Story Points")
    
    print("\nStory Points Distribution:")
    print(df_clean['Story Points'].value_counts().sort_index())
    
    return df_clean

def prepare_data(jira_data):

    # Combine Summary and Description
    text_features = jira_data['Summary'] + ' ' + jira_data['Description'].fillna('')
    
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(text_features)
    
    return X.toarray(), vectorizer

def analyze_distributions(story_points):
    plt.figure(figsize=(15, 10))
    
    # Normalize data for beta distribution
    story_points_normalized = (story_points - min(story_points)) / (max(story_points) - min(story_points))
    
    # Actual data histogram
    plt.subplot(2, 2, 1)
    counts, bins, _ = plt.hist(story_points, bins=30, density=True, alpha=0.6, label='Actual Data')
    
    # Fit distributions
    x = np.linspace(0, max(story_points), 100)
    x_norm = np.linspace(0, 1, 100)
    fits = []
    
    def calculate_aic(y, y_pred):
        n = len(y)
        residual = np.sum((y - y_pred) ** 2)
        k = 2
        aic = n * np.log(residual/n) + 2*k
        return aic
    
    params = stats.gamma.fit(story_points)
    alpha_gamma, loc_gamma, scale_gamma = params
    
    gamma_fitted = stats.gamma.pdf(x, alpha_gamma, loc=loc_gamma, scale=scale_gamma)
    plt.plot(x, gamma_fitted, 'r-', label='Gamma')
    
    ks_gamma = stats.kstest(story_points, 'gamma', args=params)
    gamma_pred = stats.gamma.pdf(bins[:-1], alpha_gamma, loc=loc_gamma, scale=scale_gamma)
    gamma_aic = calculate_aic(counts, gamma_pred)
    
    fits.append(DistributionFit(
        type=DistributionType.GAMMA,
        params=(alpha_gamma, 1/scale_gamma),  
        ks_statistic=ks_gamma.statistic,
        aic=gamma_aic
    ))
    
    # Lognormal 
    shape_lognorm, loc_lognorm, scale_lognorm = stats.lognorm.fit(story_points)
    lognorm_fitted = stats.lognorm.pdf(x, shape_lognorm, loc=loc_lognorm, scale=scale_lognorm)
    plt.plot(x, lognorm_fitted, 'g-', label='Log-normal')
    
    lognorm_pred = stats.lognorm.pdf(bins[:-1], shape_lognorm, loc=loc_lognorm, scale=scale_lognorm)
    lognorm_aic = calculate_aic(counts, lognorm_pred)
    
    fits.append(DistributionFit(
        type=DistributionType.LOGNORMAL,
        params=(np.log(scale_lognorm), shape_lognorm),
        ks_statistic=ks_gamma.statistic,
        aic=lognorm_aic
    ))
    
    # Beta
    a_beta, b_beta, loc_beta, scale_beta = stats.beta.fit(story_points_normalized)
    beta_fitted = stats.beta.pdf(x_norm, a_beta, b_beta)
    plt.plot(x_norm * max(story_points), beta_fitted / max(story_points), 'm-', label='Beta')
    
    beta_pred = stats.beta.pdf(np.linspace(0, 1, len(bins)-1), a_beta, b_beta)
    beta_aic = calculate_aic(counts, beta_pred)
    
    fits.append(DistributionFit(
        type=DistributionType.BETA,
        params=(a_beta, b_beta),
        ks_statistic=ks_gamma.statistic,
        aic=beta_aic
    ))
    
    plt.title('Distribution Fitting Comparison')
    plt.xlabel('Story Points')
    plt.ylabel('Density')
    plt.legend()
    
    # Q-Q plots
    plt.subplot(2, 2, 2)
    stats.probplot(story_points, dist="gamma", sparams=(alpha_gamma,), plot=plt)
    plt.title('Gamma Q-Q Plot')
    
    plt.subplot(2, 2, 3)
    stats.probplot(story_points, dist="lognorm", sparams=(shape_lognorm,), plot=plt)
    plt.title('Log-normal Q-Q Plot')
    
    plt.subplot(2, 2, 4)
    stats.probplot(story_points_normalized, dist="beta", sparams=(a_beta, b_beta), plot=plt)
    plt.title('Beta Q-Q Plot')
    
    plt.tight_layout()
    plt.show()
    
    print("\nFit Quality Metrics:")
    for fit in fits:
        print(f"\n{fit.type.name}:")
        print(f"AIC = {fit.aic:.3f}")
        print(f"KS statistic = {fit.ks_statistic:.3f}")
        
        if fit.type == DistributionType.BETA:
            test_data = story_points_normalized
            dist_name = 'beta'
        elif fit.type == DistributionType.LOGNORMAL:
            test_data = story_points
            dist_name = 'lognorm'
        else:  
            test_data = story_points
            dist_name = 'gamma'
            
        try:
            ks_result = stats.kstest(test_data, dist_name, args=fit.params)
            print(f"KS p-value = {ks_result.pvalue:.3e}")
        except Exception as e:
            print(f"Warning: KS test failed for {fit.type.name}: {str(e)}")
    
    print(f"""
Distribution Parameters:
    Gamma: α={alpha_gamma:.2f}, β={1/scale_gamma:.2f}
    LogNorm: σ={shape_lognorm:.2f}, μ={np.log(scale_lognorm):.2f}
    Beta: a={a_beta:.2f}, b={b_beta:.2f}

Summary Statistics:
    Mean: {np.mean(story_points):.2f}
    Median: {np.median(story_points):.2f}
    Std Dev: {np.std(story_points):.2f}
    Skewness: {stats.skew(story_points):.2f}

Unique values: {len(np.unique(story_points))}
    """)
    
    fits.sort(key=lambda x: x.aic)
    
    return fits

def make_predictions(trace, X):
    weights = trace['weights'].mean(axis=0)
    intercept = trace['intercept'].mean()
    return np.exp(intercept + np.dot(X, weights))

def model_with_jeffreys_prior(X, y):
    with pm.Model() as model:
        weights = pm.Normal('weights', mu=0, sigma=10, shape=X.shape[1])
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        mu = intercept + pm.math.dot(X, weights)
        
        sigma = pm.HalfCauchy('sigma', beta=1)
        
        y_obs = pm.Lognormal('y_obs', mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(10000, tune=1000, return_inferencedata=False)
        
    return model, trace

def model_with_prior(X, y, dist_type, params):
    with pm.Model() as model:
        weights = pm.Normal('weights', mu=0, sigma=1, shape=X.shape[1])
        intercept = pm.Normal('intercept', mu=np.log(np.mean(y)), sigma=1)
        
        mu = intercept + pm.math.dot(X, weights)
        
        if dist_type == DistributionType.GAMMA:
            alpha, beta = params
            sigma = pm.HalfNormal('sigma', sigma=1)
            y_obs = pm.Gamma('y_obs', alpha=alpha, beta=beta/pm.math.exp(mu), observed=y)
        
        elif dist_type == DistributionType.LOGNORMAL:
            mu_prior, sigma_prior = params
            sigma = pm.HalfNormal('sigma', sigma=sigma_prior)
            y_obs = pm.Lognormal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        elif dist_type == DistributionType.WEIBULL:
            c, scale = params
            sigma = pm.HalfNormal('sigma', sigma=1)
            y_obs = pm.Weibull('y_obs', alpha=c, beta=scale*pm.math.exp(mu), observed=y)
        
        elif dist_type == DistributionType.BETA:
            a, b = params
            y_norm = (y - y.min()) / (y.max() - y.min())
            sigma = pm.HalfNormal('sigma', sigma=1)
            y_obs = pm.Beta('y_obs', alpha=a, beta=b, observed=y_norm)
        
        trace = pm.sample(10000, tune=1000, return_inferencedata=False)
        
    return model, trace

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    weights = y_true / np.sum(y_true)
    wmape = np.sum(weights * np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'wmape': wmape,
        'smape': smape
    }    

def make_predictions_with_cap(hist_trace, jeff_trace, X, cap=8):
    preds_hist = make_predictions(hist_trace, X)
    preds_jeff = make_predictions(jeff_trace, X)
    
    preds_hist_capped = np.minimum(preds_hist, cap)
    preds_jeff_capped = np.minimum(preds_jeff, cap)
    
    return preds_hist_capped, preds_jeff_capped

def train_capped_model(X, y, best_fit, cap=8):
    mask = y <= cap
    X_capped = X[mask]
    y_capped = y[mask]
    
    hist_model, hist_trace = model_with_prior(X_capped, y_capped, best_fit.type, best_fit.params)
    jeff_model, jeff_trace = model_with_jeffreys_prior(X_capped, y_capped)
    
    return hist_model, hist_trace, jeff_model, jeff_trace

def plot_prediction_analysis(y_test, uncapped_results, capped_results):
    plt.figure(figsize=(15, 10))
    
    # Uncapped 
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, uncapped_results['predictions']['hist'], 
               alpha=0.5, label='Historical Prior')
    plt.scatter(y_test, uncapped_results['predictions']['jeff'], 
               alpha=0.5, label='Jeffreys Prior')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', label='Perfect Prediction')
    
    plt.xlabel('Actual Story Points')
    plt.ylabel('Predicted Story Points')
    plt.title('Uncapped Model Predictions')
    plt.legend()
    
    # Capped 
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, capped_results['predictions']['hist'], 
               alpha=0.5, label='Historical Prior')
    plt.scatter(y_test, capped_results['predictions']['jeff'], 
               alpha=0.5, label='Jeffreys Prior')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'k--', label='Perfect Prediction')
    plt.axhline(y=8, color='r', linestyle='--', label='Cap Line')
    
    plt.xlabel('Actual Story Points')
    plt.ylabel('Predicted Story Points')
    plt.title('Capped Model Predictions')
    plt.legend()
    
    # Prediction 
    plt.subplot(2, 2, 3)
    plt.hist(uncapped_results['predictions']['hist'], alpha=0.5, 
            label='Uncapped', bins=30)
    plt.hist(capped_results['predictions']['hist'], alpha=0.5, 
            label='Capped', bins=30)
    plt.xlabel('Predicted Story Points')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_distribution_comparison(y_train, best_fit, uncapped_results, capped_results):
    plt.figure(figsize=(20, 10))
    
    # Uncapped
    plt.subplot(2, 2, 1)
    x = np.linspace(0, max(y_train) * 1.5, 1000)
    
    if best_fit.type == DistributionType.LOGNORMAL:
        mu, sigma = best_fit.params
        historical_prior = stats.lognorm.pdf(x, sigma, scale=np.exp(mu))
        plt.plot(x, historical_prior, 'b-', label='Historical Prior')
        
    plt.hist(y_train, density=True, alpha=0.3, bins=30, label='Training Data')
    plt.xlabel('Story Points')
    plt.ylabel('Density')
    plt.title('Uncapped: Historical Prior vs Data')
    plt.legend()
    
    # Capped 
    plt.subplot(2, 2, 2)
    y_train_capped = np.minimum(y_train, 8)
    x_capped = np.linspace(0, 8, 1000)
    
    if best_fit.type == DistributionType.LOGNORMAL:
        shape_capped, loc_capped, scale_capped = stats.lognorm.fit(y_train_capped)
        historical_prior_capped = stats.lognorm.pdf(x_capped, shape_capped, 
                                                  loc=loc_capped, scale=scale_capped)
        plt.plot(x_capped, historical_prior_capped, 'b-', label='Historical Prior (Capped)')
        
    plt.hist(y_train_capped, density=True, alpha=0.3, bins=30, label='Training Data (Capped)')
    plt.xlabel('Story Points')
    plt.ylabel('Density')
    plt.title('Capped (≤8): Historical Prior vs Data')
    plt.legend()
    
    # Posterior
    plt.subplot(2, 2, 3)
    weights_hist_uncapped = uncapped_results['hist_trace']['weights']
    weights_jeff_uncapped = uncapped_results['jeff_trace']['weights']
    
    plt.hist(weights_hist_uncapped[:, 0], bins=30, alpha=0.5, 
             label='Historical Prior', density=True)
    plt.hist(weights_jeff_uncapped[:, 0], bins=30, alpha=0.5, 
             label='Jeffreys Prior', density=True)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Uncapped: Posterior Weight Distributions')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    weights_hist_capped = capped_results['hist_trace']['weights']
    weights_jeff_capped = capped_results['jeff_trace']['weights']
    
    plt.hist(weights_hist_capped[:, 0], bins=30, alpha=0.5, 
             label='Historical Prior', density=True)
    plt.hist(weights_jeff_capped[:, 0], bins=30, alpha=0.5, 
             label='Jeffreys Prior', density=True)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('Capped: Posterior Weight Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_analysis_data(y_test, uncapped_results, capped_results):
    unique_points, point_counts = np.unique(y_test, return_counts=True)
    print("\nAll unique story points and their counts:")
    for sp, count in zip(unique_points, point_counts):
        print(f"Story Points {sp}: {count} samples")
    
    error_data = []
    confidence_data = []
    
    for sp in unique_points:  
        mask = y_test == sp
        n_samples = np.sum(mask)
        
        if n_samples > 0:
            actual = y_test[mask]
            uncapped_preds = uncapped_results['predictions']['hist'][mask]
            capped_preds = np.minimum(capped_results['predictions']['hist'][mask], 8)
            
            uncapped_mae = np.mean(np.abs(uncapped_preds - actual))
            capped_mae = np.mean(np.abs(capped_preds - actual))
            
            error_data.append({
                'story_points': sp,
                'uncapped_mae': uncapped_mae,
                'capped_mae': capped_mae,
                'count': n_samples
            })
            
            confidence_data.append({
                'story_points': sp,
                'uncapped_mean': np.mean(uncapped_preds),
                'uncapped_lower': np.percentile(uncapped_preds, 2.5),
                'uncapped_upper': np.percentile(uncapped_preds, 97.5),
                'capped_mean': np.mean(capped_preds),
                'capped_lower': np.percentile(capped_preds, 2.5),
                'capped_upper': np.percentile(capped_preds, 97.5),
                'count': n_samples
            })
    
    error_df = pd.DataFrame(error_data)
    confidence_df = pd.DataFrame(confidence_data)
    
    print("\nNumber of story points found:", len(error_df))
    print("Story points in error_df:", error_df['story_points'].tolist())
    print("Counts in error_df:", error_df['count'].tolist())
    
    return error_df, confidence_df

def print_performance_metrics(y_test, uncapped_results, capped_results):
    print("\nUncapped Models:")
    print("Historical Prior:")
    print(f"RMSE: {uncapped_results['metrics']['hist']['rmse']:.2f} story points")
    print(f"MAE: {uncapped_results['metrics']['hist']['mae']:.2f} story points")
    print(f"Weighted MAPE: {uncapped_results['metrics']['hist']['wmape']:.2f}%")
    
    print("\nJeffreys Prior:")
    print(f"RMSE: {uncapped_results['metrics']['jeff']['rmse']:.2f} story points")
    print(f"MAE: {uncapped_results['metrics']['jeff']['mae']:.2f} story points")
    print(f"Weighted MAPE: {uncapped_results['metrics']['jeff']['wmape']:.2f}%")
    
    print("\nCapped Models:")
    print("Historical Prior:")
    print(f"RMSE: {capped_results['metrics']['hist']['rmse']:.2f} story points")
    print(f"MAE: {capped_results['metrics']['hist']['mae']:.2f} story points")
    print(f"Weighted MAPE: {capped_results['metrics']['hist']['wmape']:.2f}%")
    
    print("\nJeffreys Prior:")
    print(f"RMSE: {capped_results['metrics']['jeff']['rmse']:.2f} story points")
    print(f"MAE: {capped_results['metrics']['jeff']['mae']:.2f} story points")
    print(f"Weighted MAPE: {capped_results['metrics']['jeff']['wmape']:.2f}%")
    
    print("\nError Breakdown by Story Point Size:")
    point_sizes = np.sort(np.unique(y_test))
    for points in point_sizes:
        mask = y_test == points
        if np.sum(mask) > 0:
            print(f"\nStory Points = {points:.1f} (n={np.sum(mask)}):")
            
            # Uncapped
            hist_mae = np.mean(np.abs(uncapped_results['predictions']['hist'][mask] - y_test[mask]))
            jeff_mae = np.mean(np.abs(uncapped_results['predictions']['jeff'][mask] - y_test[mask]))
            print("Uncapped Models:")
            print(f"  Historical MAE: {hist_mae:.2f}")
            print(f"  Jeffreys MAE: {jeff_mae:.2f}")
            
            # Capped
            hist_mae_capped = np.mean(np.abs(capped_results['predictions']['hist'][mask] - y_test[mask]))
            jeff_mae_capped = np.mean(np.abs(capped_results['predictions']['jeff'][mask] - y_test[mask]))
            print("Capped Models:")
            print(f"  Historical MAE: {hist_mae_capped:.2f}")
            print(f"  Jeffreys MAE: {jeff_mae_capped:.2f}")        

def plot_analysis(error_df, confidence_df):
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 1, 1)
    width = 0.35
    x = np.arange(len(error_df))
    
    plt.bar(x - width/2, error_df['uncapped_mae'], width, label='Uncapped Model')
    plt.bar(x + width/2, error_df['capped_mae'], width, label='Capped Model')
    
    for i, row in enumerate(error_df.itertuples()):
        plt.text(i, max(row.uncapped_mae, row.capped_mae), f'n={row.count}',
                ha='center', va='bottom')
    
    plt.xlabel('Story Points')
    plt.ylabel('Mean Absolute Error')
    plt.title('Prediction Error by Story Point Size')
    plt.xticks(x, [str(sp) for sp in error_df['story_points']], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    
    story_points = confidence_df['story_points'].values
    
    print("Plot Story Points:", story_points)
    
    # Uncapped
    plt.plot(story_points, confidence_df['uncapped_mean'], 
             'b-o', label='Uncapped Mean', zorder=3)
    plt.fill_between(story_points,
                     confidence_df['uncapped_lower'],
                     confidence_df['uncapped_upper'],
                     alpha=0.2, color='blue', zorder=1)
    
    # Capped
    plt.plot(story_points, confidence_df['capped_mean'], 
             'r-o', label='Capped Mean', zorder=2)
    plt.fill_between(story_points,
                     confidence_df['capped_lower'],
                     confidence_df['capped_upper'],
                     alpha=0.2, color='red', zorder=1)
    
    plt.axhline(y=8, color='r', linestyle='--', alpha=0.5, label='Cap Line')
    
    plt.xlabel('Story Points')
    plt.ylabel('Predicted Story Points')
    plt.title('Prediction Confidence Intervals')
    plt.legend()
    
    plt.xscale('linear')
    plt.grid(True, alpha=0.3)
    
    plt.xlim(min(story_points), max(story_points) + 1)
    plt.ylim(0, max(max(confidence_df['uncapped_upper']), 8 * 1.1))
    
    plt.xticks(story_points, [str(x) for x in story_points])
    
    plt.tight_layout()
    plt.show()

def plot_predictions_with_ci(y_test, preds_hist, preds_jeff, title, cap=None):
    plt.scatter(y_test, preds_hist, alpha=0.5, label='Historical Prior')
    plt.scatter(y_test, preds_jeff, alpha=0.5, label='Jeffreys Prior')
    
    max_val = cap if cap else max(max(y_test), max(preds_hist), max(preds_jeff))
    plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction')
    
    if cap:
        plt.plot([0, max_val], [cap, cap], 'r--', label='Cap Line')
        plt.yticks(np.arange(0, cap + 1, 1))
    
    plt.xticks(np.arange(0, int(max_val) + 1, 1))
    plt.xlabel('Actual Story Points')
    plt.ylabel('Predicted Story Points')
    plt.title(title)
    plt.legend()

def plot_error_distribution(y_test, preds_hist, preds_jeff, title):
    plt.hist(preds_hist - y_test, alpha=0.5, label='Historical Prior', bins=20)
    plt.hist(preds_jeff - y_test, alpha=0.5, label='Jeffreys Prior', bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

def plot_prediction_comparison(y_test, uncapped_results, capped_results):
    plt.figure(figsize=(20, 10))
    
    # Uncapped
    plt.subplot(2, 2, 1)
    plot_predictions_with_ci(y_test, 
                           uncapped_results['predictions']['hist'],
                           uncapped_results['predictions']['jeff'],
                           title='Uncapped Model Predictions')
    
    # Capped
    plt.subplot(2, 2, 2)
    plot_predictions_with_ci(y_test, 
                           capped_results['predictions']['hist'],
                           capped_results['predictions']['jeff'],
                           title='Capped Model Predictions',
                           cap=8)
    
    # Errors
    plt.subplot(2, 2, 3)
    plot_error_distribution(y_test, 
                          uncapped_results['predictions']['hist'],
                          uncapped_results['predictions']['jeff'],
                          title='Uncapped Model Errors')
    
    plt.subplot(2, 2, 4)
    plot_error_distribution(y_test, 
                          capped_results['predictions']['hist'],
                          capped_results['predictions']['jeff'],
                          title='Capped Model Errors')
    
    plt.tight_layout()
    plt.show()
        
def run_inference(model_results, vectorizer, input_file, output_file, n_rows=100):
    print(f"Loading first {n_rows} rows from {input_file}...")
    input_data = pd.read_csv(input_file, nrows=n_rows)
    
    text_features = input_data['Summary'] + ' ' + input_data['Description'].fillna('')
    X_new = vectorizer.transform(text_features).toarray()
    
    print("Making predictions...")
    pred = make_predictions(model_results['hist_trace'], X_new)
    pred_capped = np.minimum(pred, 8)
    
    output_data = input_data[['Summary', 'Description']].copy()
    output_data['Story Points'] = pred_capped.round(1)
    
    print(f"Saving predictions to {output_file}...")
    output_data.to_csv(output_file, index=False)
    
    print("\nPrediction Summary:")
    print(output_data['Story Points'].describe())
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(output_data['Story Points'], bins=20, alpha=0.7)
    plt.xlabel('Predicted Story Points')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Story Points')
    plt.grid(True, alpha=0.3)
    
    sorted_predictions = np.sort(pred_capped)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(sorted_predictions)), sorted_predictions, 'b-', label='Prediction')
    plt.axhline(y=8, color='r', linestyle='--', label='Cap Line')
    plt.xlabel('Story Index (sorted by prediction)')
    plt.ylabel('Story Points')
    plt.title('Sorted Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return output_data

def main():
    print("Loading and preparing data...")
    jira_data = pd.read_csv('data/jira_data_with_story_points.csv')
    clean_data = clean_story_points(jira_data)
    X, vectorizer = prepare_data(clean_data)
    y = clean_data['Story Points'].values

    print(f"\nInitial data distribution:")
    print(pd.Series(y).value_counts().sort_index())
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTest set distribution:")
    print(pd.Series(y_test).value_counts().sort_index())
    
    print("\nAnalyzing distributions...")
    fits = analyze_distributions(y)
    best_fit = fits[0]
    print(f"Best fitting distribution: {best_fit.type.name}")
    
    print("\nTraining uncapped models...")
    hist_model, hist_trace = model_with_prior(X_train, y_train, best_fit.type, best_fit.params)
    jeff_model, jeff_trace = model_with_jeffreys_prior(X_train, y_train)
    
    preds_hist = make_predictions(hist_trace, X_test)
    preds_jeff = make_predictions(jeff_trace, X_test)
    
    uncapped_results = {
        'hist_trace': hist_trace,
        'jeff_trace': jeff_trace,
        'predictions': {
            'hist': preds_hist,
            'jeff': preds_jeff
        },
        'metrics': {
            'hist': calculate_metrics(y_test, preds_hist),
            'jeff': calculate_metrics(y_test, preds_jeff)
        }
    }
    
    print("\nTraining capped models (≤ 8 points)...")
    hist_model_capped, hist_trace_capped, jeff_model_capped, jeff_trace_capped = \
        train_capped_model(X_train, y_train, best_fit, cap=8)
    
    preds_hist_capped, preds_jeff_capped = make_predictions_with_cap(
        hist_trace_capped, jeff_trace_capped, X_test)
    
    capped_results = {
        'hist_trace': hist_trace_capped,
        'jeff_trace': jeff_trace_capped,
        'predictions': {
            'hist': preds_hist_capped,
            'jeff': preds_jeff_capped
        },
        'metrics': {
            'hist': calculate_metrics(y_test, preds_hist_capped),
            'jeff': calculate_metrics(y_test, preds_jeff_capped)
        }
    }
    
    print("\nGenerating visualizations...")
    
    plot_distribution_comparison(y_train, best_fit, uncapped_results, capped_results)
    plot_prediction_comparison(y_test, uncapped_results, capped_results)
    
    print("\nUnique story points in test set:", np.sort(np.unique(y_test)))
    print("Test set shape:", y_test.shape)
    print("Predictions shape:", preds_hist.shape)

    error_df, confidence_df = create_analysis_data(y_test, uncapped_results, capped_results)
    plot_analysis(error_df, confidence_df)
    
    print("\nStory points in error_df:", np.sort(error_df['story_points'].unique()))
    print("Number of rows in error_df:", len(error_df))

    print("\nCalculating performance metrics...")
    print_performance_metrics(y_test, uncapped_results, capped_results)
    
    print("\nRunning inference on new data...")
    run_inference(capped_results, vectorizer, 
                 'data/jira_data_without_story_points.csv',
                 'story_point_inference.csv')


if __name__ == "__main__":
    main()