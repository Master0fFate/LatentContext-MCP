import { v4 as uuidv4 } from "uuid";
import {
    upsertEntity,
    getEntityByLabel,
    getEntityById,
    searchEntities,
    upsertRelation,
    getRelationsBySubject,
    getRelationsByObject,
    getRelationsByPredicate,
    getAllEntities,
    getAllRelations,
    getEntityCount,
    getRelationCount,
    deleteEntity,
    deleteRelation,
    getDb,
    saveDatabase,
    type EntityRow,
    type RelationRow,
} from "./database.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GraphFact {
    subject: string;
    subjectType: string;
    predicate: string;
    object: string;
    objectType: string;
    confidence: number;
    temporalStart: string | null;
}

export interface GraphQueryResult {
    entity: EntityRow;
    outgoing: RelationRow[];
    incoming: RelationRow[];
    neighbors: EntityRow[];
    serialized: string;
}

// ---------------------------------------------------------------------------
// Entity management
// ---------------------------------------------------------------------------

/**
 * Ensure an entity exists in the knowledge graph, creating it if needed.
 * Returns the entity ID.
 */
export function ensureEntity(
    label: string,
    entityType: string = "unknown",
    properties: Record<string, unknown> = {},
    confidence: number = 1.0
): string {
    // Check if entity already exists by label (case-insensitive)
    const existing = getEntityByLabel(label);
    if (existing) {
        // Update confidence if new one is higher
        if (confidence > existing.confidence) {
            upsertEntity(existing.id, label, entityType, properties, confidence);
        }
        return existing.id;
    }

    const id = uuidv4();
    upsertEntity(id, label, entityType, properties, confidence);
    return id;
}

// ---------------------------------------------------------------------------
// Fact storage
// ---------------------------------------------------------------------------

/**
 * Store a fact as a subject-predicate-object triple in the knowledge graph.
 * Handles entity creation/lookup and conflict resolution automatically.
 */
export function storeFact(
    subjectLabel: string,
    predicate: string,
    objectLabel: string,
    subjectType: string = "unknown",
    objectType: string = "unknown",
    confidence: number = 1.0,
    properties: Record<string, unknown> = {}
): string {
    const subjectId = ensureEntity(subjectLabel, subjectType, {}, confidence);
    const objectId = ensureEntity(objectLabel, objectType, {}, confidence);

    const relationId = uuidv4();
    upsertRelation(relationId, subjectId, predicate, objectId, properties, confidence);

    return relationId;
}

// ---------------------------------------------------------------------------
// Graph queries
// ---------------------------------------------------------------------------

/**
 * Query the knowledge graph for all facts about a specific entity.
 * Supports multi-hop traversal via the `depth` parameter.
 */
export function queryEntity(
    entityLabel: string,
    depth: number = 1
): GraphQueryResult | null {
    const entity = getEntityByLabel(entityLabel);
    if (!entity) {
        // Try fuzzy search
        const matches = searchEntities(entityLabel);
        if (matches.length === 0) return null;
        return queryEntityById(matches[0].id, depth);
    }
    return queryEntityById(entity.id, depth);
}

/**
 * Query by entity ID with BFS traversal up to `depth` hops.
 */
export function queryEntityById(
    entityId: string,
    depth: number = 1
): GraphQueryResult | null {
    const entity = getEntityById(entityId);
    if (!entity) return null;

    const outgoing = getRelationsBySubject(entityId);
    const incoming = getRelationsByObject(entityId);

    // Collect neighbor entity IDs
    const neighborIds = new Set<string>();
    for (const rel of outgoing) neighborIds.add(rel.object_id);
    for (const rel of incoming) neighborIds.add(rel.subject_id);

    const neighbors: EntityRow[] = [];
    for (const nId of neighborIds) {
        const n = getEntityById(nId);
        if (n) neighbors.push(n);
    }

    // If depth > 1, recursively collect deeper neighbors
    if (depth > 1) {
        const deeperRelations: RelationRow[] = [];
        const deeperNeighborIds = new Set<string>();
        for (const neighbor of neighbors) {
            const nOutgoing = getRelationsBySubject(neighbor.id);
            const nIncoming = getRelationsByObject(neighbor.id);
            for (const rel of [...nOutgoing, ...nIncoming]) {
                const targetId =
                    rel.subject_id === neighbor.id ? rel.object_id : rel.subject_id;
                if (targetId !== entityId && !neighborIds.has(targetId)) {
                    deeperNeighborIds.add(targetId);
                    deeperRelations.push(rel);
                }
            }
        }
        for (const dId of deeperNeighborIds) {
            const deepNeighbor = getEntityById(dId);
            if (deepNeighbor) neighbors.push(deepNeighbor);
        }
        outgoing.push(...deeperRelations.filter((r) => r.subject_id === entityId || neighborIds.has(r.subject_id)));
        incoming.push(...deeperRelations.filter((r) => r.object_id === entityId || neighborIds.has(r.object_id)));
    }

    const serialized = serializeQueryResult(entity, outgoing, incoming, neighbors);

    return { entity, outgoing, incoming, neighbors, serialized };
}

/**
 * Query relations by predicate type.
 */
export function queryByPredicate(predicate: string): GraphFact[] {
    const relations = getRelationsByPredicate(predicate);
    const facts: GraphFact[] = [];

    for (const rel of relations) {
        const subject = getEntityById(rel.subject_id);
        const object = getEntityById(rel.object_id);
        if (subject && object) {
            facts.push({
                subject: subject.label,
                subjectType: subject.entity_type,
                predicate: rel.predicate,
                object: object.label,
                objectType: object.entity_type,
                confidence: rel.confidence,
                temporalStart: rel.temporal_start,
            });
        }
    }

    return facts;
}

/**
 * Get the full graph schema — entity types and relation types currently in the graph.
 */
export function getGraphSchema(): { entityTypes: string[]; predicateTypes: string[] } {
    const entities = getAllEntities();
    const relations = getAllRelations();

    const entityTypes = [...new Set(entities.map((e) => e.entity_type))];
    const predicateTypes = [...new Set(relations.map((r) => r.predicate))];

    return { entityTypes, predicateTypes };
}

/**
 * Get all facts as a flat list of triples.
 */
export function getAllFacts(): GraphFact[] {
    const relations = getAllRelations();
    const facts: GraphFact[] = [];

    for (const rel of relations) {
        const subject = getEntityById(rel.subject_id);
        const object = getEntityById(rel.object_id);
        if (subject && object) {
            facts.push({
                subject: subject.label,
                subjectType: subject.entity_type,
                predicate: rel.predicate,
                object: object.label,
                objectType: object.entity_type,
                confidence: rel.confidence,
                temporalStart: rel.temporal_start,
            });
        }
    }

    return facts;
}

// ---------------------------------------------------------------------------
// Graph modification
// ---------------------------------------------------------------------------

/**
 * Remove an entity and all its relations from the knowledge graph.
 */
export function removeEntity(entityLabel: string): boolean {
    const entity = getEntityByLabel(entityLabel);
    if (!entity) return false;

    // Delete all relations involving this entity
    const outgoing = getRelationsBySubject(entity.id);
    const incoming = getRelationsByObject(entity.id);
    for (const rel of [...outgoing, ...incoming]) {
        deleteRelation(rel.id);
    }

    deleteEntity(entity.id);
    return true;
}

/**
 * Update a specific relation's confidence or properties.
 */
export function deprecateRelation(relationId: string, newConfidence: number = 0.1): boolean {
    try {
        const db = getDb();
        db.run(
            "UPDATE relations SET confidence = ?, temporal_end = ? WHERE id = ?",
            [newConfidence, new Date().toISOString(), relationId]
        );
        saveDatabase();
        return true;
    } catch {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/**
 * Serialize a query result into a compact, human-readable text block.
 * This is what gets injected into the LLM context — conciseness is critical.
 */
function serializeQueryResult(
    entity: EntityRow,
    outgoing: RelationRow[],
    incoming: RelationRow[],
    neighbors: EntityRow[]
): string {
    const lines: string[] = [];

    lines.push(`Entity: ${entity.label} (${entity.entity_type})`);

    const neighborMap = new Map(neighbors.map((n) => [n.id, n]));

    // Outgoing relations: entity → predicate → object
    for (const rel of outgoing) {
        const obj = neighborMap.get(rel.object_id);
        if (obj) {
            const conf = rel.confidence < 1.0 ? ` [conf:${rel.confidence.toFixed(2)}]` : "";
            lines.push(`  → ${rel.predicate} → ${obj.label}${conf}`);
        }
    }

    // Incoming relations: subject → predicate → entity
    for (const rel of incoming) {
        const subj = neighborMap.get(rel.subject_id);
        if (subj) {
            const conf = rel.confidence < 1.0 ? ` [conf:${rel.confidence.toFixed(2)}]` : "";
            lines.push(`  ← ${subj.label} → ${rel.predicate}${conf}`);
        }
    }

    return lines.join("\n");
}

/**
 * Serialize a list of facts into a compact text block.
 */
export function serializeFacts(facts: GraphFact[]): string {
    if (facts.length === 0) return "No facts found.";

    return facts
        .map((f) => {
            const conf = f.confidence < 1.0 ? ` [conf:${f.confidence.toFixed(2)}]` : "";
            return `${f.subject} → ${f.predicate} → ${f.object}${conf}`;
        })
        .join("\n");
}

/**
 * Get graph statistics.
 */
export function getGraphStats(): { entities: number; relations: number } {
    return {
        entities: getEntityCount(),
        relations: getRelationCount(),
    };
}
