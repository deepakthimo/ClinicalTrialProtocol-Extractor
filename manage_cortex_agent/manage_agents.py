import os
import re
import json
import sys
import time
import requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to sys.path so we can import core.config
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.config import PAGE_AGENT_COPIES, SEQUENTIAL_AGENT_COPIES

# Set your Auth Cookie or Token here
CORTEX_COOKIE = os.getenv("CORTEX_COOKIE", None)
BASE_URL = "https://cortex.lilly.com/model"

# Owner email — each team member sets their own in .env
OWNER_EMAIL = os.getenv("OWNER_EMAIL", None)


def _get_owner_prefix() -> str:
    """Extract a short, URL-safe prefix from OWNER_EMAIL.
    e.g. 'deepak.tm@network.lilly.com' -> 'deepaktm'
    """
    if not OWNER_EMAIL:
        raise SystemExit(
            "ERROR: OWNER_EMAIL is not set in .env.\n"
            "Each team member must set OWNER_EMAIL=<your-lilly-email> before deploying agents."
        )
    local_part = OWNER_EMAIL.split("@")[0]          # 'deepak.tm'
    return re.sub(r"[^a-z0-9]", "", local_part.lower())  # 'deepaktm'

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "cookie": CORTEX_COOKIE
}

def get_base_payload():
    """Returns the massive boilerplate payload with default keys."""
    return {
        "name": "", 
        "displayName": "",
        "model_description": "",
        "auth": {
            "owners": [],
            "owners_group": [],
            "owners_aws_roles": [],
            "users": [],
            "access_groups": [],
            "access_aws_roles": [],
            "allow_access_to_reports_of": [],
            "private": False
        },
        "security_config": None,
        "chain": [{
            "chain_class": "model-only-chain",
            "model_iteration": 1,
            "order": 1,
            "chain_params": {}
        }],
        "model_versions": [{
            "model_class": "",
            "model_iteration": 1,
            "priority": 1,
            "reasoning_effort": None,
            "enable_thinking": False,
            "advanced_param_overrides": {}
        }],
        "prompts": {
            "no_context": "default_with_no_context",
            "with_context": "default_with_context",
            "with_json_context": "default_with_json_context",
            "enhance_query": "default_enhance_query",
            "sql": "default_sql",
            "agent_tool": "default_agent_tool",
            "cortex_agent_tool_prompt_v2": "default_cortex_agent_tool_prompt_v2",
            "cortex_agent_tool_prompt_v2_with_file_handling": "default_cortex_agent_tool_prompt_v2_with_file_handling",
            "cortex_agent_action": "default_cortex_agent_action",
            "cortex_agent_reasoning": "default_cortex_agent_reasoning",
            "realtime_agent": "default_realtime_agent",
            "table_summary": "default_table_summary",
            "summary": "default_summary",
            "entity_extraction": "default_entity_extraction",
            "rewrite": "default_rewrite",
            "kg_triple_extraction": "default_kg_triple_extraction"
        },
        "toolkits": [],
        "allowed_tools_list": [],
        "data": [],
        "max_response_token_size": 40960,
        "doc_relevence_threshold": 0.7,
        "hybrid_search": {
            "rrf_relevance_threshold": 0.7,
            "rrf_constant": 60,
            "hybrid_score_type": "rrf",
            "lexical_average_weight": 0.5
        },
        "agent_tool_max_iterations": 10,
        "app_binding": "chatbuilder",
        "multimodal": False,
        "labels": {},
        "k_value": 5,
        "token_buffer_size": 1000,
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "stop": [],
        "seed": 0,
        "logprobs": False,
        "rerank": {
            "enabled": False,
            "model": "cohere-rerank-3.5",
            "top_n": 20
        },
        "document_limit_to_search": 0,
        "session_config": None,
        "sync_a2a_card": False,
        "context_cache_key": ""
    }

def build_agent_names(base_name: str, num_copies: int = 3, owner_prefix: str = ""):
    """Generates N copies for concurrency: base, copy1, ... copy(N-1).
    If owner_prefix is provided, the final deployed name becomes '{owner}-{base}'.
    """
    prefixed = f"{owner_prefix}-{base_name}" if owner_prefix else base_name
    names = [prefixed]
    for i in range(1, num_copies):
        names.append(f"{prefixed}-copy{i}")
    return names

def get_num_copies(agent: dict) -> int:
    """Derive the number of copies from core.config.
    Page agents scale with concurrency; master/crop stay fixed.
    An explicit 'num_copies' in agents_config.json overrides the auto-calculation.
    """
    if "num_copies" in agent:
        return agent["num_copies"]
    if agent.get("labels", {}).get("agent") == "page":
        return PAGE_AGENT_COPIES
    return SEQUENTIAL_AGENT_COPIES

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = str(_SCRIPT_DIR / "agents_config.json")


def should_enable_thinking(model_class, model_iteration) -> bool:
    """Enable thinking only for Claude models with iteration > 14."""
    if not isinstance(model_class, str):
        return False

    try:
        iteration = int(model_iteration)
    except (TypeError, ValueError):
        return False

    return model_class.strip().lower() == "claude" and iteration > 14


def deploy_agents(config_file=None):
    """Reads JSON and POSTs configurations to create agents."""
    config_file = config_file or _DEFAULT_CONFIG
    owner_prefix = _get_owner_prefix()

    with open(config_file, "r") as f:
        config = json.load(f)

    # Inject owner email into auth (from env, not hardcoded in JSON)
    global_owners = config["global_auth"].get("owners", [])
    if OWNER_EMAIL and OWNER_EMAIL not in global_owners:
        global_owners = [OWNER_EMAIL] + global_owners
    global_users = config["global_auth"].get("users", [])
    global_labels = config.get("global_labels", {})

    print("🚀 STARTING BULK AGENT DEPLOYMENT...\n")

    for agent in config["agents"]:
        num_copies = get_num_copies(agent)
        names_to_create = build_agent_names(agent["base_name"], num_copies, owner_prefix)
        agent_specific_labels = agent.get("labels", {})
        chain_class = agent.get("chain_class", "model-only-chain")
        
        for idx, name in enumerate(names_to_create):
            payload = get_base_payload()
            
            # Inject dynamic values
            payload["name"] = name
            payload["displayName"] = name
            payload["model_description"] = agent["description"]
            payload["multimodal"] = agent["multimodal"]

            payload["chain"][0]["chain_class"] = chain_class
            
            # If it is an agent-chain, the API supports extra parameters
            if chain_class == "agent-chain":
                payload["chain"][0]["max_turns"] = agent.get("max_turns", 10)
                payload["chain"][0]["react_enabled"] = True
                # WE REMOVED THE PROMPT OVERRIDE HERE. Let the API use its default for agent-chains.
            else:
                # Standard model-only-chain prompt injection
                payload["prompts"]["no_context"] = agent.get("no_context_prompt", "default_with_no_context")
            
            # Inject Model class & iteration
            payload["model_versions"][0]["model_class"] = agent["model_class"]
            payload["model_versions"][0]["model_iteration"] = agent["model_iteration"]
            payload["model_versions"][0]["enable_thinking"] = should_enable_thinking(
                agent.get("model_class"),
                agent.get("model_iteration"),
            )
            
            # Inject custom prompt template
            payload["prompts"]["no_context"] = agent["no_context_prompt"]
            
            # Inject Auth
            payload["auth"]["owners"] = global_owners
            payload["auth"]["users"] = global_users

            # Labelling - Handling Metadata
            merged_labels = global_labels.copy()
            merged_labels.update(agent_specific_labels)
            #Add replica ID (base, copy1, copy2) for concurrency debugging
            replica_id = "base" if idx == 0 else f"copy{idx}"
            merged_labels["replica_id"] = replica_id

            payload["labels"] = merged_labels

            time.sleep(2.5)
            print(f"Creating: {name} ... ", end="")
            response = requests.post(BASE_URL, headers=HEADERS, json=payload)
            
            if response.status_code == 200:
                print("✅ SUCCESS")
            else:
                print(f"❌ FAILED ({response.status_code}) -> {response.text}")

def delete_agents(config_file=None):
    """Reads JSON and sends DELETE requests to clean up all agents and their copies."""
    config_file = config_file or _DEFAULT_CONFIG
    owner_prefix = _get_owner_prefix()

    with open(config_file, "r") as f:
        config = json.load(f)

    print("🗑️ STARTING BULK AGENT DELETION...\n")

    for agent in config["agents"]:
        num_copies = get_num_copies(agent)
        names_to_delete = build_agent_names(agent["base_name"], num_copies, owner_prefix)
        
        for name in names_to_delete:
            url = f"{BASE_URL}/{name}"
            print(f"Deleting: {name} ... ", end="")
            
            response = requests.delete(url, headers=HEADERS)
            
            if response.status_code == 200:
                print("✅ DELETED")
            elif response.status_code == 404:
                print("⚠️ NOT FOUND (Already deleted)")
            else:
                print(f"❌ FAILED ({response.status_code}) -> {response.text}")

def generate_registry(config_file=None):
    """Auto-generate agents/agent_registry.py from agents_config.json."""
    owner_prefix = _get_owner_prefix()
    config_file = config_file or _DEFAULT_CONFIG
    with open(config_file, "r") as f:
        config = json.load(f)

    registry = {}
    for agent in config["agents"]:
        num_copies = get_num_copies(agent)
        registry[agent["base_name"]] = build_agent_names(agent["base_name"], num_copies, owner_prefix)

    # Write the registry file
    registry_path = Path(__file__).resolve().parent.parent / "agents" / "agent_registry.py"
    lines = [
        "# AUTO-GENERATED by manage_agents.py — do not edit manually",
        "",
        "AGENT_REGISTRY: dict[str, list[str]] = {",
    ]
    for base_name, names in registry.items():
        lines.append(f"    {base_name!r}: {names!r},")
    lines.append("}")
    lines.append("")  # trailing newline

    registry_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Registry written to {registry_path} ({len(registry)} agents)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manage Cortex Agents")
    parser.add_argument("action", choices=["deploy", "delete", "registry"],
                        help="Action to perform: 'deploy', 'delete', or 'registry' (regenerate agent_registry.py)")
    args = parser.parse_args()

    if args.action == "registry":
        generate_registry()
    elif args.action == "deploy":
        deploy_agents()
    elif args.action == "delete":
        # Double check before mass deletion
        confirm = input("Are you sure you want to DELETE all agents in the config? (y/n): ")
        if confirm.lower() == 'y':
            delete_agents()
        else:
            print("Deletion cancelled.")