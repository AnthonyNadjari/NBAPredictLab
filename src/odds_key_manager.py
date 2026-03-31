"""
Simple API Key Manager for The Odds API
Stores multiple keys and selects the best one based on quota
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import requests


class OddsKeyManager:
    """Manages multiple Odds API keys and selects best available"""

    def __init__(self, keys_file: str = "data/odds_api_keys.json"):
        """
        Initialize key manager

        Args:
            keys_file: Path to JSON file storing API keys
        """
        self.keys_file = Path(keys_file)
        self.keys: Dict[str, str] = {}
        self._load_keys()

    def _load_keys(self):
        """Load keys from file and environment"""
        # Load from file if exists
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    self.keys = json.load(f)
            except Exception as e:
                print(f"Error loading keys file: {e}")
                self.keys = {}

        # Also check environment variable
        env_key = os.getenv('ODDS_API_KEY')
        if env_key and env_key not in self.keys.values():
            # Add env key with default name if not already present
            self.keys['default'] = env_key

    def _save_keys(self):
        """Save keys to file"""
        try:
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            print(f"Error saving keys: {e}")

    def add_key(self, name: str, api_key: str):
        """
        Add a new API key

        Args:
            name: Friendly name for the key (e.g., "key1", "backup_key")
            api_key: The actual API key
        """
        self.keys[name] = api_key
        self._save_keys()

    def remove_key(self, name: str):
        """Remove an API key by name"""
        if name in self.keys:
            del self.keys[name]
            self._save_keys()

    def get_key_quota(self, api_key: str) -> Dict:
        """
        Check quota for a specific API key

        Returns:
            {
                'remaining': int,
                'used': int,
                'status': 'OK' or 'ERROR',
                'error': str (if error)
            }
        """
        try:
            response = requests.get(
                "https://api.the-odds-api.com/v4/sports",
                params={'apiKey': api_key},
                timeout=5
            )

            if response.status_code == 200:
                remaining = response.headers.get('x-requests-remaining', '0')
                used = response.headers.get('x-requests-used', '0')

                return {
                    'remaining': int(remaining),
                    'used': int(used),
                    'status': 'OK'
                }
            elif response.status_code == 401:
                return {'error': 'Invalid API key', 'status': 'ERROR', 'remaining': 0, 'used': 0}
            else:
                return {'error': f'HTTP {response.status_code}', 'status': 'ERROR', 'remaining': 0, 'used': 0}

        except Exception as e:
            return {'error': str(e), 'status': 'ERROR', 'remaining': 0, 'used': 0}

    def get_all_quotas(self) -> Dict[str, Dict]:
        """
        Get quota info for all stored keys

        Returns:
            {
                'key_name': {
                    'api_key': 'masked_key',
                    'remaining': int,
                    'used': int,
                    'status': 'OK' or 'ERROR'
                },
                ...
            }
        """
        quotas = {}

        for name, api_key in self.keys.items():
            quota_info = self.get_key_quota(api_key)
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

            quotas[name] = {
                'api_key': masked_key,
                'full_key': api_key,
                **quota_info
            }

        return quotas

    def get_best_key(self, min_remaining: int = 50) -> Optional[str]:
        """
        Get the best API key with sufficient quota

        Args:
            min_remaining: Minimum requests remaining required (default: 50)

        Returns:
            API key string or None if no suitable key found
        """
        best_key = None
        max_remaining = 0

        for name, api_key in self.keys.items():
            quota = self.get_key_quota(api_key)

            if quota.get('status') == 'OK':
                remaining = quota.get('remaining', 0)

                # Check if this key meets minimum and has more remaining than current best
                if remaining >= min_remaining and remaining > max_remaining:
                    best_key = api_key
                    max_remaining = remaining

        return best_key

    def get_key_by_name(self, name: str) -> Optional[str]:
        """Get a specific key by name"""
        return self.keys.get(name)

    def list_keys(self) -> List[str]:
        """Get list of all key names"""
        return list(self.keys.keys())


# Convenience functions for easy usage
def get_best_odds_api_key(min_remaining: int = 50) -> Optional[str]:
    """
    Get the best available Odds API key with sufficient quota

    Args:
        min_remaining: Minimum requests remaining (default: 50)

    Returns:
        API key string or None
    """
    manager = OddsKeyManager()
    return manager.get_best_key(min_remaining)


def check_all_key_quotas() -> Dict[str, Dict]:
    """
    Check quotas for all stored API keys

    Returns:
        Dictionary with key names and their quota info
    """
    manager = OddsKeyManager()
    return manager.get_all_quotas()


if __name__ == "__main__":
    # Fix Windows encoding
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # Test the manager
    manager = OddsKeyManager()

    print("=" * 60)
    print("ODDS API KEY MANAGER TEST")
    print("=" * 60)
    print()

    print("All Keys and Quotas:")
    print("-" * 60)
    quotas = manager.get_all_quotas()

    for name, info in quotas.items():
        print(f"\n{name}:")
        print(f"  Key: {info['api_key']}")
        print(f"  Status: {info['status']}")
        if info['status'] == 'OK':
            print(f"  Remaining: {info['remaining']}")
            print(f"  Used: {info['used']}")
        else:
            print(f"  Error: {info.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("BEST KEY SELECTION (min 50 remaining):")
    print("=" * 60)

    best_key = manager.get_best_key(min_remaining=50)
    if best_key:
        masked = best_key[:8] + "..." + best_key[-4:]
        print(f"[OK] Best key: {masked}")
        quota = manager.get_key_quota(best_key)
        print(f"  Remaining: {quota.get('remaining', 'Unknown')}")
    else:
        print("[WARNING] No key with 50+ remaining requests found")
