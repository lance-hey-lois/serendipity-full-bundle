import React, { useState, useEffect } from 'react';
import { SearchResult } from '../types';

interface ResultCardProps {
  result: SearchResult;
  index: number;
  streamingText?: string;
}

const ResultCard: React.FC<ResultCardProps> = ({ result, index, streamingText }) => {
  const [isRemoving, setIsRemoving] = useState(false);
  const [shouldHide, setShouldHide] = useState(false);
  
  // Debug logging
  if (streamingText) {
    console.log(`ResultCard ${index} received streaming text:`, streamingText);
  }

  useEffect(() => {
    if (result.status === 'rejected') {
      // Start fade-out animation
      setIsRemoving(true);
      // After animation completes, hide the element
      const timer = setTimeout(() => {
        setShouldHide(true);
      }, 500); // 0.5s duration
      return () => clearTimeout(timer);
    }
  }, [result.status]);

  if (shouldHide) {
    return null; // Remove from DOM completely
  }

  const getStatusClasses = () => {
    if (isRemoving) {
      return 'opacity-0 scale-95 max-h-0 mb-0 p-0 border-0 overflow-hidden';
    }
    
    switch (result.status) {
      case 'validated':
        return 'border-venture-green bg-venture-green/10 animate-slide-in max-h-96';
      default:
        return 'border-venture-border opacity-90 max-h-96';
    }
  };

  return (
    <div 
      className={`relative bg-venture-card/80 backdrop-blur border-2 rounded-xl p-6 mb-4 transition-all duration-500 ${getStatusClasses()}`}
      style={{
        transitionProperty: 'opacity, transform, max-height, margin-bottom, padding, border-width',
      }}>
      {/* Score Badge */}
      <div className="absolute top-4 right-4 bg-venture-accent text-white px-3 py-1 rounded-full font-bold text-sm">
        Score: {(result.quantumScore * 100).toFixed(0)}%
      </div>

      {/* Profile Info */}
      <div className="mb-4">
        <h3 className="text-venture-accent text-xl font-bold mb-2">
          {index + 1}. {result.name}
        </h3>
        <p className="text-venture-text">
          {result.title} at {result.company}
        </p>
      </div>

      {/* Skills Pills */}
      <div className="flex flex-wrap gap-2 mb-4">
        {result.skills.map((skill, idx) => (
          <span
            key={idx}
            className="bg-venture-accent/20 text-venture-accent px-3 py-1 rounded-full text-sm border border-venture-accent/30"
          >
            {skill}
          </span>
        ))}
      </div>

      {/* Explanation */}
      {(streamingText || result.explanation) && (
        <div className="mt-4 p-4 bg-black/20 border-l-4 border-venture-accent rounded italic text-venture-light">
          {streamingText || result.explanation}
        </div>
      )}

      {/* Loading Indicator */}
      {result.status === 'pending' && !streamingText && (
        <div className="mt-4 p-4 bg-black/20 border-l-4 border-venture-accent rounded italic text-venture-light/70">
          ⏳ Validating match...
        </div>
      )}
    </div>
  );
};

export default ResultCard;