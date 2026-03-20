from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntityResult:
    name: str
    affinity: float
    popularity: float
    tags: list[str]
    description: str
    address: str = ""


@dataclass
class DomainResult:
    city: str
    domain: str
    domain_label: str
    results: list[EntityResult]
    error: Optional[str] = None
