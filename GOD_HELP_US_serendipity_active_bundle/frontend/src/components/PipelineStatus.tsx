import React from 'react';
import { PipelineStatus as PipelineStatusType } from '../types';

interface PipelineStatusProps {
  status: PipelineStatusType;
  times: {
    embeddings?: number;
    quantum?: number;
    display?: number;
    validation?: number;
    total?: number;
  };
}

const PipelineStatus: React.FC<PipelineStatusProps> = ({ status, times }) => {
  const steps = [
    { key: 'embeddings', icon: '📡', label: 'Embeddings' },
    { key: 'quantum', icon: '⚛️', label: 'Quantum' },
    { key: 'display', icon: '📊', label: 'Display' },
    { key: 'validation', icon: '🤖', label: 'Validation' },
  ];

  const getStepClasses = (stepKey: string) => {
    if (times[stepKey as keyof typeof times]) {
      return 'bg-venture-green/20 text-venture-green';
    }
    if (status.phase === stepKey) {
      return 'bg-venture-accent text-white animate-pulse-scale';
    }
    return 'bg-black/20 text-venture-light/50';
  };

  if (status.phase === 'complete') {
    return (
      <div className="bg-black/20 backdrop-blur p-4 rounded-lg text-center mb-6">
        <span className="inline-block px-4 py-2 rounded-full bg-venture-green/20 text-venture-green">
          ✅ Complete Pipeline: {times.total?.toFixed(2)}s
          {status.validatedCount !== undefined && ` | ${status.validatedCount} matches found`}
        </span>
      </div>
    );
  }

  return (
    <div className="bg-black/20 backdrop-blur p-4 rounded-lg text-center mb-6">
      <div className="flex justify-center gap-2 flex-wrap">
        {steps.map((step) => (
          <span
            key={step.key}
            className={`inline-block px-4 py-2 rounded-full transition-all duration-300 ${getStepClasses(step.key)}`}
          >
            {times[step.key as keyof typeof times] ? (
              <>✅ {step.label} ({times[step.key as keyof typeof times]?.toFixed(2)}s)</>
            ) : (
              <>{step.icon} {step.label}</>
            )}
          </span>
        ))}
      </div>
      {status.message && (
        <p className="text-venture-light/70 text-sm mt-2">{status.message}</p>
      )}
    </div>
  );
};

export default PipelineStatus;