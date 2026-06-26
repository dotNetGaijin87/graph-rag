// Shapes mirror the backend's JSON serializers.

export interface IngestionReport {
  document_id: string;
  title: string;
  chunk_count: number;
  entity_count: number;
  relationship_count: number;
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  entities: string[];
}

export interface GraphFact {
  source: string;
  type: string;
  target: string;
  description: string;
  sentence: string;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  context: {
    chunks: RetrievedChunk[];
    facts: GraphFact[];
  };
}

export interface Stats {
  documents: number;
  chunks: number;
  entities: number;
  relationships: number;
}

export interface GraphNode {
  id: string;
  type: string;
  description: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  description: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Settings {
  // editable
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  enable_entity_extraction: boolean;
  max_extraction_chars: number;
  // read-only (informational)
  llm_model: string;
  embedding_model: string;
  embedding_dim: number;
}

export type EditableSettings = Pick<
  Settings,
  'chunk_size' | 'chunk_overlap' | 'top_k' | 'enable_entity_extraction' | 'max_extraction_chars'
>;
