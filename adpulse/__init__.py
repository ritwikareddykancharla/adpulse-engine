"""AdPulse: Sponsored Catalog Ranking, GSP Budget Pacing, Off-Policy Evaluation & Seller Diagnostics."""

from .catalog_ranker import (
    CatalogSKU,
    LexicalBidRanker,
    RankedAdCandidate,
    RankerBenchmarkReport,
    ShopperQuery,
    TwoStageAdRanker,
    generate_synthetic_catalog_and_queries,
    run_ranker_benchmark,
)
from .auction_pacer import (
    AdvertiserCampaign,
    AuctionOpportunity,
    BudgetPacedAuctionEngine,
    PacingComparisonReport,
    SimulationMetrics,
    generate_auction_workload,
    run_pacing_benchmark,
)
from .off_policy_eval import (
    BootstrapInterval,
    EstimatorAccuracySummary,
    LoggedAuctionEvent,
    OPEBenchmarkReport,
    OffPolicyBidEvaluator,
    PolicyEvaluationResult,
    generate_logged_propensity_traces,
    run_ope_benchmark,
)
from .campaign_diagnostics import (
    CampaignDiagnosticInput,
    CampaignDiagnosticReport,
    CampaignDiagnostics,
    DiagnosticsBenchmarkReport,
    IssueCode,
    generate_diagnostic_scenarios,
    run_diagnostics_benchmark,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CatalogSKU",
    "ShopperQuery",
    "RankedAdCandidate",
    "LexicalBidRanker",
    "TwoStageAdRanker",
    "RankerBenchmarkReport",
    "generate_synthetic_catalog_and_queries",
    "run_ranker_benchmark",
    "AdvertiserCampaign",
    "AuctionOpportunity",
    "BudgetPacedAuctionEngine",
    "SimulationMetrics",
    "PacingComparisonReport",
    "generate_auction_workload",
    "run_pacing_benchmark",
    "LoggedAuctionEvent",
    "BootstrapInterval",
    "PolicyEvaluationResult",
    "EstimatorAccuracySummary",
    "OPEBenchmarkReport",
    "OffPolicyBidEvaluator",
    "generate_logged_propensity_traces",
    "run_ope_benchmark",
    "IssueCode",
    "CampaignDiagnosticInput",
    "CampaignDiagnosticReport",
    "CampaignDiagnostics",
    "DiagnosticsBenchmarkReport",
    "generate_diagnostic_scenarios",
    "run_diagnostics_benchmark",
]
