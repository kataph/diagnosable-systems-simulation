"""
Knowledge-graph representation of a diagnosable physical system.

Every system is a labeled directed graph whose nodes are *entities* and
whose edges are *typed relations*.  Only a specific subset of relation
types (ELECTRICALLY_CONNECTED) is relevant to electrical simulation;
the others capture structural and physical knowledge independently.

Entity types
------------
COMPONENT   A physical component (holds a ``Component`` object).
            Modules are simply components that have PART_OF edges pointing
            to them (typically ``PhysicalEnclosure`` objects).

Relation types
--------------
PART_OF
    ``from`` is a functional part of ``to`` (a COMPONENT acting as a module,
    e.g. a ``PhysicalEnclosure``).
    Parthood is functional: a component belongs to a module by design
    role, independently of where it is physically placed.
    E.g. psu_cable_pos PART_OF cube_psu.

CONTAINED_IN
    ``from`` is physically inside the enclosure ``to`` (a COMPONENT of
    type PhysicalEnclosure).
    Containment is spatial: a component sits inside a box.
    E.g. psu_green_led CONTAINED_IN cube_psu.

    Note: parthood and containment are independent.  A cable that exits
    a cube is still *part of* the module it belongs to (PART_OF), even
    though it is not *contained in* the cube (no CONTAINED_IN edge).

ELECTRICALLY_CONNECTED
    A port of ``from`` is directly wired to a port of ``to``.
    Stored as: from_id = component_id, to_id = component_id,
    attrs["from_port"] = port name on the source component,
    attrs["to_port"]   = port name on the target component,
    attrs["is_ground"] = True (optional) marks that the shared net is
                         the voltage reference (0 V).
    Electrical nodes are NOT stored in the KG; they are derived by
    ``build_circuit_from_kg`` via union-find over these edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EntityType(Enum):
    COMPONENT = auto()   # a physical component (Component object)


class RelationType(Enum):
    PART_OF               = auto()  # functional membership in a module
    CONTAINED_IN          = auto()  # physical containment inside an enclosure
    ELECTRICALLY_CONNECTED = auto() # port of component wired to an electrical node


@dataclass
class KGEdge:
    from_id:  str
    to_id:    str
    relation: RelationType
    attrs:    dict[str, Any] = field(default_factory=dict)


class SystemGraph:
    """
    A labeled directed graph of entities and typed relations.

    Building a graph
    ----------------
    - Call ``add_entity()`` for every component and every module.
    - Call ``add_edge()`` to declare typed relations between entities.

    Querying
    --------
    - ``get_entity(id)``            → the object stored for that entity.
    - ``entities_of_type(type)``    → all entities of a given type.
    - ``outgoing(from_id, rel)``    → edges leaving an entity (optionally filtered).
    - ``incoming(to_id, rel)``      → edges entering an entity (optionally filtered).
    """

    def __init__(self) -> None:
        self._entities: dict[str, tuple[EntityType, Any]] = {}
        self._edges: list[KGEdge] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        entity_type: EntityType,
        obj: Any = None,
    ) -> None:
        if entity_id in self._entities:
            raise ValueError(f"Entity {entity_id!r} already exists.")
        self._entities[entity_id] = (entity_type, obj)

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: RelationType,
        **attrs: Any,
    ) -> None:
        self._edges.append(
            KGEdge(from_id=from_id, to_id=to_id, relation=relation, attrs=attrs)
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Any:
        try:
            return self._entities[entity_id][1]
        except KeyError:
            raise KeyError(
                f"No entity {entity_id!r} in graph. "
                f"Known: {list(self._entities)}"
            )

    def entity_type(self, entity_id: str) -> EntityType:
        return self._entities[entity_id][0]

    def entities_of_type(self, entity_type: EntityType) -> dict[str, Any]:
        return {
            eid: obj
            for eid, (etype, obj) in self._entities.items()
            if etype == entity_type
        }

    def edges_of_relation(self, relation: RelationType) -> list[KGEdge]:
        """All edges of a given relation type."""
        return [e for e in self._edges if e.relation == relation]

    def outgoing(
        self,
        from_id: str,
        relation: RelationType | None = None,
    ) -> list[KGEdge]:
        return [
            e for e in self._edges
            if e.from_id == from_id
            and (relation is None or e.relation == relation)
        ]

    def incoming(
        self,
        to_id: str,
        relation: RelationType | None = None,
    ) -> list[KGEdge]:
        return [
            e for e in self._edges
            if e.to_id == to_id
            and (relation is None or e.relation == relation)
        ]

    def __repr__(self) -> str:
        return (
            f"SystemGraph(entities={len(self._entities)}, "
            f"edges={len(self._edges)})"
        )
