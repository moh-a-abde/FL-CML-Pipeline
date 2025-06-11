"""
Random Forest specific federated learning strategies.

This module implements federated learning strategies optimized for Random Forest models,
including parameter aggregation and evaluation strategies suitable for ensemble methods.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class RandomForestFedAvg(fl.server.strategy.FedAvg):
    """
    Federated Averaging strategy adapted for Random Forest models.
    
    This strategy implements federated averaging for Random Forest models,
    focusing on aggregating feature importances and ensemble predictions.
    """
    
    def __init__(self, 
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2):
        """
        Initialize Random Forest FedAvg strategy.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients
        )
        
        logger.info("Initialized RandomForestFedAvg strategy")
        logger.info("Fraction fit: %s, Fraction evaluate: %s", fraction_fit, fraction_evaluate)
        logger.info("Min fit clients: %d, Min evaluate clients: %d", min_fit_clients, min_evaluate_clients)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results from multiple clients.
        
        For Random Forest, this aggregates feature importances and other
        ensemble-specific parameters.
        
        Args:
            server_round: Current server round
            results: Fit results from clients
            failures: Failed fit attempts
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        logger.info("Aggregating Random Forest parameters from %d clients in round %d", 
                   len(results), server_round)
        
        # Log any failures
        if failures:
            logger.warning("Round %d had %d failures", server_round, len(failures))
        
        # Extract parameters and weights from results
        parameters_list = []
        weights = []
        
        for _, fit_res in results:
            if fit_res.parameters:
                parameters_list.append(parameters_to_ndarrays(fit_res.parameters))
                weights.append(fit_res.num_examples)
        
        if not parameters_list:
            logger.warning("No parameters received from clients in round %d", server_round)
            return None, {}
        
        # Aggregate feature importances using weighted average
        aggregated_parameters = self._aggregate_random_forest_parameters(
            parameters_list, weights
        )
        
        # Calculate aggregated metrics
        total_examples = sum(weight for weight in weights)
        
        # Aggregate training metrics
        aggregated_metrics = {}
        if results and results[0][1].metrics:
            metric_keys = results[0][1].metrics.keys()
            for key in metric_keys:
                weighted_sum = sum(
                    fit_res.metrics.get(key, 0) * fit_res.num_examples
                    for _, fit_res in results
                    if fit_res.metrics and key in fit_res.metrics
                )
                aggregated_metrics[f"train_{key}"] = weighted_sum / total_examples
        
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(results)
        
        logger.info("Round %d aggregation completed. Total examples: %d, Participating clients: %d", 
                   server_round, total_examples, len(results))
        
        return ndarrays_to_parameters(aggregated_parameters), aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from multiple clients.
        
        Args:
            server_round: Current server round
            results: Evaluation results from clients
            failures: Failed evaluation attempts
            
        Returns:
            Aggregated loss and metrics
        """
        if not results:
            return None, {}
        
        logger.info("Aggregating evaluation results from %d clients in round %d", 
                   len(results), server_round)
        
        # Log any failures
        if failures:
            logger.warning("Round %d evaluation had %d failures", server_round, len(failures))
        
        # Weighted average of losses and metrics
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        
        # Aggregate loss
        weighted_loss_sum = sum(
            eval_res.loss * eval_res.num_examples for _, eval_res in results
        )
        aggregated_loss = weighted_loss_sum / total_examples
        
        # Aggregate metrics
        aggregated_metrics = {}
        if results and results[0][1].metrics:
            metric_keys = results[0][1].metrics.keys()
            for key in metric_keys:
                weighted_sum = sum(
                    eval_res.metrics.get(key, 0) * eval_res.num_examples
                    for _, eval_res in results
                    if eval_res.metrics and key in eval_res.metrics
                )
                aggregated_metrics[key] = weighted_sum / total_examples
        
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["num_clients"] = len(results)
        
        logger.info("Round %d evaluation completed. Avg loss: %.4f, Total examples: %d", 
                   server_round, aggregated_loss, total_examples)
        
        return aggregated_loss, aggregated_metrics
    
    def _aggregate_random_forest_parameters(self, 
                                          parameters_list: List[List[np.ndarray]], 
                                          weights: List[int]) -> List[np.ndarray]:
        """
        Aggregate Random Forest parameters (feature importances) using weighted average.
        
        Args:
            parameters_list: List of parameter arrays from each client
            weights: Number of examples used by each client
            
        Returns:
            Aggregated parameters
        """
        if not parameters_list:
            return []
        
        # Check if all clients have the same parameter structure
        if not all(len(params) == len(parameters_list[0]) for params in parameters_list):
            logger.warning("Inconsistent parameter structure across clients")
            return parameters_list[0]  # Return first client's parameters as fallback
        
        total_weight = sum(weights)
        aggregated_params = []
        
        # Aggregate each parameter array
        for i in range(len(parameters_list[0])):
            param_arrays = [params[i] for params in parameters_list]
            
            # Check if all arrays have the same shape
            if not all(arr.shape == param_arrays[0].shape for arr in param_arrays):
                logger.warning("Inconsistent parameter shape for parameter %d", i)
                aggregated_params.append(param_arrays[0])  # Use first client's parameters
                continue
            
            # Weighted average
            weighted_sum = sum(
                weight * param_array 
                for weight, param_array in zip(weights, param_arrays)
            )
            aggregated_param = weighted_sum / total_weight
            aggregated_params.append(aggregated_param)
        
        logger.info("Aggregated %d parameter arrays", len(aggregated_params))
        return aggregated_params


class RandomForestBagging(RandomForestFedAvg):
    """
    Bagging strategy for Random Forest federated learning.
    
    This strategy implements a bagging approach where each client trains
    independent Random Forest models and the server aggregates predictions.
    """
    
    def __init__(self, **kwargs):
        """Initialize Random Forest Bagging strategy."""
        super().__init__(**kwargs)
        logger.info("Initialized RandomForestBagging strategy")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using bagging approach.
        
        In bagging, each client's model is treated as an independent estimator
        in the ensemble, so we primarily aggregate statistics rather than
        model parameters.
        """
        if not results:
            return None, {}
        
        logger.info("Bagging aggregation from %d clients in round %d", 
                   len(results), server_round)
        
        # For bagging, we don't necessarily need to aggregate parameters
        # Instead, we can maintain separate models and aggregate predictions
        
        # Calculate ensemble statistics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        
        # Aggregate training metrics
        aggregated_metrics = {}
        if results and results[0][1].metrics:
            metric_keys = results[0][1].metrics.keys()
            for key in metric_keys:
                values = [
                    fit_res.metrics.get(key, 0)
                    for _, fit_res in results
                    if fit_res.metrics and key in fit_res.metrics
                ]
                if values:
                    aggregated_metrics[f"avg_{key}"] = np.mean(values)
                    aggregated_metrics[f"std_{key}"] = np.std(values)
        
        aggregated_metrics["total_examples"] = total_examples
        aggregated_metrics["ensemble_size"] = len(results)
        
        # Return empty parameters since we're using bagging
        empty_params = ndarrays_to_parameters([])
        
        logger.info("Bagging round %d completed. Ensemble size: %d", 
                   server_round, len(results))
        
        return empty_params, aggregated_metrics


def create_random_forest_strategy(strategy_name: str = "fedavg", **kwargs) -> Strategy:
    """
    Factory function to create Random Forest federated learning strategies.
    
    Args:
        strategy_name: Name of the strategy ('fedavg', 'bagging')
        **kwargs: Additional strategy parameters
        
    Returns:
        Strategy: Configured federated learning strategy
    """
    strategy_map = {
        "fedavg": RandomForestFedAvg,
        "bagging": RandomForestBagging,
    }
    
    if strategy_name.lower() not in strategy_map:
        logger.warning("Unknown strategy '%s', using fedavg", strategy_name)
        strategy_name = "fedavg"
    
    strategy_class = strategy_map[strategy_name.lower()]
    
    logger.info("Creating Random Forest strategy: %s", strategy_name)
    return strategy_class(**kwargs) 