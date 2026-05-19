"""
PromptShield-MCP Configuration
"""

# Risk thresholds (τ₁, τ₂)
TAU_1 = 0.35   # Below this → Accept
TAU_2 = 0.70   # Above this → Reject, between → Repair

# Iterative loop settings
MAX_ITERATIONS = 3
CONVERGENCE_THRESHOLD = 0.05

# Feature weights (logistic regression w vector)
# F(P) = [override, injection, role_manipulation, context_manipulation, ambiguity]
FEATURE_WEIGHTS = [2.1, 1.8, 1.6, 1.4, 0.9]
BIAS = -1.2

# Utility metric weights
ALPHA = 0.4   # helpfulness
BETA  = 0.4   # faithfulness
GAMMA = 0.2   # cost penalty

# MCP authorized tools
AUTHORIZED_TOOLS = ["search_web", "retrieve_document", "run_python"]

# Attack categories (for dataset)
ATTACK_CATEGORIES = [
    "prompt_injection",
    "instruction_override",
    "role_confusion",
    "context_manipulation",
    "policy_bypass",
    "tool_coercion",
]

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# LLM mock mode (set False when using real API)
MOCK_LLM = False

OPENAI_MODEL = "gpt-4o-mini"   # swap to your key's model

GROQ_API_KEY  = __import__('os').environ.get('GROQ_API_KEY', '')
GROQ_BASE_URL  = 'https://api.groq.com/openai/v1'
GROQ_MODEL     = 'llama-3.3-70b-versatile'
