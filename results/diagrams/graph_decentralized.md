# Architecture Decentralized — handoffs Command(goto=...), 0 superviseur LLM

> **Note** : Le routage `Command(goto=...)` est dynamique — invisible au `draw_mermaid()` statique.
> Ce diagramme est reconstruit manuellement depuis les règles `_ROUTING_RULES` dans `decentralized.py`.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	code_quality(code_quality)
	security(security)
	license(license)
	community(community)
	documentation(documentation)
	synthesize(synthesize)
	__end__([<p>__end__</p>]):::last
	__start__ --> code_quality;
	code_quality -.->|"HIGH/CRIT dep findings"| security;
	code_quality -.->|otherwise| community;
	security --> license;
	license --> community;
	community --> documentation;
	documentation --> synthesize;
	synthesize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
