import React from 'react';

interface SerendipityToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

const SerendipityToggle: React.FC<SerendipityToggleProps> = ({ enabled, onChange }) => {
  return (
    <div className="serendipity-toggle">
      <label className="toggle-switch">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="slider">
          <span className="toggle-label">
            {enabled ? '🌌 Quantum Serendipity' : '🔍 Classic Search'}
          </span>
        </span>
      </label>
      <div className="toggle-description">
        {enabled 
          ? 'Discover unexpected connections through quantum patterns'
          : 'Traditional similarity-based search'
        }
      </div>
    </div>
  );
};

export default SerendipityToggle;