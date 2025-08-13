export interface User {
  userId: string;
  name: string;
}

export interface SearchResult {
  index: number;
  name: string;
  title: string;
  company: string;
  skills: string[];
  quantumScore: number;
  status: 'pending' | 'validated' | 'rejected';
  explanation?: string;
}

export interface PipelineStatus {
  phase: 'idle' | 'embeddings' | 'quantum' | 'display' | 'validation' | 'complete';
  message?: string;
  time?: number;
  totalTime?: number;
  validatedCount?: number;
}

export interface SSEMessage {
  type: 'status' | 'result' | 'validation_start' | 'explanation_update' | 'validation_complete' | 'complete' | 'error';
  phase?: string;
  message?: string;
  time?: number;
  index?: number;
  name?: string;
  title?: string;
  company?: string;
  skills?: string[];
  quantumScore?: number;
  status?: string;
  text?: string;
  totalTime?: number;
  validatedCount?: number;
}

export interface SerendipityResult {
  slug: string;
  name: string;
  blurb?: string;
  quantumScore: number;
  classicalScore: number;
  noveltyScore: number;
  serendipityScore: number;
  explanation: string;
  location?: string;
  company?: string;
  title?: string;
}