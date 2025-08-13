import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import ResultCard from './components/ResultCard';
import PipelineStatus from './components/PipelineStatus';
import SerendipityToggle from './components/SerendipityToggle';
import SerendipityDisplay from './components/SerendipityDisplay';
import { User, SearchResult, PipelineStatus as PipelineStatusType, SSEMessage, SerendipityResult } from './types';
import { SSEClient } from './utils/sse';

function App() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchDepth, setSearchDepth] = useState<string>('ALL Public Profiles');
  const [resultLimit, setResultLimit] = useState<number>(10);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [streamingTexts, setStreamingTexts] = useState<{ [key: number]: string }>({});
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatusType>({ phase: 'idle' });
  const [pipelineTimes, setPipelineTimes] = useState<any>({});
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const sseClientRef = useRef<SSEClient | null>(null);
  
  // Serendipity mode state
  const [serendipityEnabled, setSerendipityEnabled] = useState<boolean>(false);
  const [serendipityResults, setSerendipityResults] = useState<SerendipityResult[]>([]);
  const [serendipityLoading, setSerendipityLoading] = useState<boolean>(false);
  const [serendipityStats, setSerendipityStats] = useState<any>(null);
  const [serendipityPerformance, setSerendipityPerformance] = useState<any>(null);

  // Fetch users on mount
  useEffect(() => {
    fetch('http://localhost:8079/api/users')
      .then(res => res.json())
      .then(data => {
        setUsers(data);
        if (data.length > 0) {
          setSelectedUser(data[0].userId);
        }
      })
      .catch(err => console.error('Failed to fetch users:', err));
  }, []);

  const handleSerendipitySearch = async () => {
    if (!searchQuery.trim()) return;
    
    setSerendipityLoading(true);
    
    try {
      const response = await fetch('http://localhost:8079/api/serendipity/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          user_id: selectedUser,
          search_depth: searchDepth,
          result_limit: resultLimit
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setSerendipityResults(data.results || []);
        setSerendipityStats(data.statistics || {});
        setSerendipityPerformance(data.performance || {});
      } else {
        console.error('Serendipity search failed:', response.statusText);
        // Fallback to mock data
        setSerendipityResults([
          {
            slug: "quantum_researcher_mock",
            name: "Dr. Sarah Chen (Mock)",
            blurb: "Quantum computing researcher with expertise in photonic systems",
            quantumScore: 0.92,
            classicalScore: 0.76,
            noveltyScore: 0.88,
            serendipityScore: 0.91,
            explanation: "Mock quantum serendipity result - high coherence detected through cross-domain pattern matching",
            location: "Stanford, CA",
            company: "Quantum Ventures Lab",
            title: "Senior Research Scientist"
          }
        ]);
      }
    } catch (error) {
      console.error('Serendipity search error:', error);
    } finally {
      setSerendipityLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    // Reset state
    setResults([]);
    setStreamingTexts({});
    setPipelineStatus({ phase: 'embeddings' });
    setPipelineTimes({});
    setIsSearching(true);
    
    // If serendipity mode is enabled, also run serendipity search
    if (serendipityEnabled) {
      handleSerendipitySearch();
    }

    // Close any existing SSE connection
    if (sseClientRef.current) {
      sseClientRef.current.close();
    }

    // Create new SSE client
    const sseClient = new SSEClient();
    sseClientRef.current = sseClient;

    // Connect to SSE endpoint
    await sseClient.connect(
      'http://localhost:8079/api/search/stream',
      {
        query: searchQuery,
        userId: selectedUser,
        searchDepth: searchDepth,
        resultLimit: resultLimit
      },
      (data: SSEMessage) => {

      switch (data.type) {
        case 'status':
          setPipelineStatus({
            phase: data.phase as any,
            message: data.message
          });
          if (data.time !== undefined && data.phase) {
            setPipelineTimes((prev: any) => ({
              ...prev,
              [data.phase!]: data.time
            }));
          }
          break;

        case 'result':
          const newResult: SearchResult = {
            index: data.index!,
            name: data.name!,
            title: data.title!,
            company: data.company!,
            skills: data.skills!,
            quantumScore: data.quantumScore!,
            status: 'pending'
          };
          setResults(prev => [...prev, newResult]);
          break;

        case 'validation_start':
          // Result validation starting
          break;

        case 'explanation_update':
          console.log('Explanation update:', data.index, data.text);
          setStreamingTexts(prev => {
            const updated = {
              ...prev,
              [data.index!]: data.text!
            };
            console.log('Updated streaming texts:', updated);
            return updated;
          });
          break;

        case 'validation_complete':
          // Update results and streaming texts together to avoid stale closures
          setStreamingTexts(prev => {
            const finalText = prev[data.index!] || '';
            
            // Update results with the final text and status
            setResults(prevResults => prevResults.map(r => {
              if (r.index === data.index) {
                console.log(`Updating result ${r.index} with status ${data.status}, explanation: "${finalText}", keeping skills:`, r.skills);
                // If there's an error but we have streaming text, keep it as explanation
                const explanation = finalText || (data.status === 'error' ? 'Validation error' : '');
                return { ...r, status: data.status as any, explanation };
              }
              return r;
            }));
            
            // Only remove streaming text if validation succeeded or was rejected properly
            // Keep it for errors so the partial text remains visible
            if (data.status !== 'error' || !prev[data.index!]) {
              const newTexts = { ...prev };
              delete newTexts[data.index!];
              return newTexts;
            }
            return prev;
          });
          break;

        case 'complete':
          setPipelineStatus({
            phase: 'complete',
            totalTime: data.totalTime,
            validatedCount: data.validatedCount
          });
          setPipelineTimes((prev: any) => ({
            ...prev,
            total: data.totalTime
          }));
          setIsSearching(false);
          break;

        case 'error':
          console.error('Search error:', data.message);
          setIsSearching(false);
          break;
      }
    },
    (error) => {
      console.error('SSE error:', error);
      setIsSearching(false);
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-venture-burgundy to-venture-dark text-venture-text">
      {/* Header */}
      <div className="text-center py-8 mb-8 bg-black/20 backdrop-blur-lg rounded-2xl mx-4 mt-4">
        <h1 className="text-5xl font-bold mb-2 text-venture-text">
          🚀 Venture Capital Deal Sourcing
        </h1>
        <p className="text-venture-light">AI-Powered Network Discovery</p>
      </div>

      <div className="container mx-auto px-4 pb-8">
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          {/* Search Settings */}
          <div className="bg-venture-card/80 backdrop-blur rounded-xl p-6 border border-venture-border">
            <h3 className="text-xl font-bold mb-4 text-venture-accent">🎯 Search Settings</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Select User</label>
              <select 
                value={selectedUser}
                onChange={(e) => setSelectedUser(e.target.value)}
                className="w-full px-3 py-2 bg-venture-dark border border-venture-border rounded-lg focus:border-venture-accent focus:outline-none text-venture-text"
              >
                {users.map(user => (
                  <option key={user.userId} value={user.userId}>
                    {user.name} ({user.userId})
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Search Depth</label>
              <div className="space-y-2">
                {['1st degree connections', '2nd degree connections', 'ALL Public Profiles'].map(depth => (
                  <label key={depth} className="flex items-center">
                    <input
                      type="radio"
                      value={depth}
                      checked={searchDepth === depth}
                      onChange={(e) => setSearchDepth(e.target.value)}
                      className="mr-2 text-venture-accent"
                    />
                    <span>{depth === 'ALL Public Profiles' ? '🌍 ' + depth : depth}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Number of Results: {resultLimit}
              </label>
              <input
                type="range"
                min="5"
                max="20"
                value={resultLimit}
                onChange={(e) => setResultLimit(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Serendipity Toggle */}
          <div className="bg-venture-card/80 backdrop-blur rounded-xl p-6 border border-venture-border">
            <h3 className="text-xl font-bold mb-4 text-venture-accent">🌌 Quantum Mode</h3>
            <SerendipityToggle 
              enabled={serendipityEnabled}
              onChange={setSerendipityEnabled}
            />
          </div>

          {/* Search Query */}
          <div className="md:col-span-2 bg-venture-card/80 backdrop-blur rounded-xl p-6 border border-venture-border">
            <h3 className="text-xl font-bold mb-4 text-venture-accent">🔍 Search Query</h3>
            
            <textarea
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Catholic Venture capitalists, AI researcher, investment partner, creative director..."
              className="w-full px-4 py-3 bg-venture-dark border border-venture-border rounded-lg focus:border-venture-accent focus:outline-none resize-none h-32 mb-4 text-venture-text placeholder-venture-light/50"
            />

            <button
              onClick={handleSearch}
              disabled={isSearching || !searchQuery.trim()}
              className={`w-full py-3 px-6 rounded-lg font-bold text-lg transition-all duration-300 ${
                isSearching || !searchQuery.trim()
                  ? 'bg-gray-600 cursor-not-allowed opacity-50'
                  : 'bg-venture-accent hover:bg-venture-green hover:scale-105 hover:shadow-xl text-white'
              }`}
            >
              {isSearching ? '⏳ Searching...' : '🚀 Launch Search'}
            </button>
          </div>
        </div>

        {/* Pipeline Status */}
        {pipelineStatus.phase !== 'idle' && (
          <PipelineStatus status={pipelineStatus} times={pipelineTimes} />
        )}

        {/* Serendipity Results */}
        {serendipityEnabled && serendipityResults.length > 0 && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold mb-6 text-venture-accent">🌌 Quantum Serendipity Results</h2>
            <SerendipityDisplay 
              results={serendipityResults}
              loading={serendipityLoading}
            />
          </div>
        )}

        {/* Regular Results */}
        {results.length > 0 && (
          <div>
            <h2 className="text-2xl font-bold mb-6 text-venture-accent">
              {serendipityEnabled ? '🔍 Classical Results' : '🎯 Search Results'}
            </h2>
            <div className="space-y-4">
              {results.map((result) => (
                <ResultCard
                  key={result.index}
                  result={result}
                  index={result.index}
                  streamingText={streamingTexts[result.index]}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="text-center py-4 text-venture-light/70 text-sm">
        🚀 Venture Capital Deal Sourcing Platform - Powered by AI
      </div>
    </div>
  );
}

export default App;