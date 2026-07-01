import React, { useState } from 'react';
import DataExplorer from './components/DataExplorer';
import Chat from './components/Chat';
import ChunkViewer from './components/ChunkViewer';

function App() {
  const [selectedChunk, setSelectedChunk] = useState(null);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar / Data Explorer */}
      <div className="w-1/3 min-w-[300px] border-r border-border flex flex-col bg-surface/50">
        <div className="p-4 border-b border-border bg-surface shadow-sm z-10">
          <h1 className="text-lg font-semibold flex items-center gap-2">
            <span className="text-primary text-xl">📚</span> RAG Explorer
          </h1>
          <p className="text-xs text-textMuted mt-1">Explore chunks and chat with context.</p>
        </div>
        <div className="flex-1 overflow-hidden p-2">
          <DataExplorer onSelectChunk={setSelectedChunk} />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative bg-background">
        <div className={`flex-1 flex flex-col h-full ${selectedChunk ? 'hidden' : ''}`}>
          <Chat onSelectChunk={setSelectedChunk} />
        </div>
        {selectedChunk && (
          <div className="flex-1 flex flex-col h-full">
            <ChunkViewer 
              chunk={selectedChunk} 
              onClose={() => setSelectedChunk(null)} 
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
