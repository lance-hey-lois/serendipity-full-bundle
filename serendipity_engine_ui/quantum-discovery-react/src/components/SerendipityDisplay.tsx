import React from 'react';

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

interface SerendipityDisplayProps {
  results: SerendipityResult[];
  loading: boolean;
}

const SerendipityDisplay: React.FC<SerendipityDisplayProps> = ({ results, loading }) => {
  const getScoreColor = (score: number): string => {
    if (score > 0.8) return '#4ade80'; // green
    if (score > 0.6) return '#60a5fa'; // blue
    if (score > 0.4) return '#fbbf24'; // yellow
    return '#f87171'; // red
  };

  const getNoveltyBadge = (novelty: number): string => {
    if (novelty > 0.5) return '🎯 DISCOVERY';
    if (novelty > 0.3) return '✨ NOVEL';
    if (novelty > 0.1) return '🌱 EMERGING';
    return '';
  };

  if (loading) {
    return (
      <div className="serendipity-loading">
        <div className="quantum-spinner"></div>
        <p>Collapsing quantum wave functions...</p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="serendipity-empty">
        <p>No serendipitous connections found. Try a different query!</p>
      </div>
    );
  }

  return (
    <div className="serendipity-results">
      {results.map((result, index) => (
        <div key={result.slug} className="serendipity-card">
          <div className="serendipity-rank">#{index + 1}</div>
          
          <div className="serendipity-header">
            <h3>{result.name}</h3>
            {result.title && result.company && (
              <p className="serendipity-subtitle">
                {result.title} at {result.company}
              </p>
            )}
            {result.location && (
              <p className="serendipity-location">📍 {result.location}</p>
            )}
          </div>

          {result.blurb && (
            <p className="serendipity-blurb">{result.blurb}...</p>
          )}

          <div className="serendipity-scores">
            <div className="score-item">
              <span className="score-label">Quantum</span>
              <span 
                className="score-value" 
                style={{ color: getScoreColor(result.quantumScore) }}
              >
                {(result.quantumScore * 100).toFixed(0)}%
              </span>
            </div>
            <div className="score-item">
              <span className="score-label">Classical</span>
              <span 
                className="score-value"
                style={{ color: getScoreColor(result.classicalScore) }}
              >
                {(result.classicalScore * 100).toFixed(0)}%
              </span>
            </div>
            <div className="score-item">
              <span className="score-label">Novelty</span>
              <span 
                className="score-value"
                style={{ color: getScoreColor(result.noveltyScore) }}
              >
                {(result.noveltyScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {getNoveltyBadge(result.noveltyScore) && (
            <div className="novelty-badge">{getNoveltyBadge(result.noveltyScore)}</div>
          )}

          <div className="serendipity-explanation">
            {result.explanation}
          </div>

          <div className="serendipity-bar">
            <div 
              className="serendipity-fill"
              style={{ 
                width: `${result.serendipityScore * 100}%`,
                background: `linear-gradient(90deg, #8b5cf6 0%, #ec4899 ${result.serendipityScore * 100}%, #06b6d4 100%)`
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

export default SerendipityDisplay;