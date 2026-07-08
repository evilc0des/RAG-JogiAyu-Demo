import React, { useState } from 'react';
import DataExplorer from './components/DataExplorer';
import DataIngestion from './components/DataIngestion';
import Chat from './components/Chat';
import ChunkViewer from './components/ChunkViewer';
import { Database, MessageSquare, Upload } from 'lucide-react';

function App() {
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [activePanel, setActivePanel] = useState('explorer');

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <div className="w-1/3 min-w-[300px] border-r border-border flex flex-col bg-surface/50">
        <div className="p-4 border-b border-border bg-surface shadow-sm z-10">
          <h1 className="text-lg font-semibold flex items-center gap-2">
            <span className="text-primary text-xl">🕉️</span> Jogi Ayu RAG
          </h1>
          <p className="text-xs text-textMuted mt-1">Ayurvedic Medicine Educational Assistant</p>
        </div>

        <div className="flex border-b border-border">
          {[
            { id: 'explorer', label: 'Explorer', icon: Database },
            { id: 'ingest', label: 'Ingest', icon: Upload },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => { setActivePanel(id); setSelectedChunk(null); }}
              className={`flex-1 py-2 text-xs font-medium transition-colors flex items-center justify-center gap-1 ${
                activePanel === id
                  ? 'text-primary border-b-2 border-primary bg-primary/5'
                  : 'text-textMuted hover:text-text'
              }`}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-hidden">
          {activePanel === 'explorer' && (
            <div className="p-2 h-full">
              <DataExplorer onSelectChunk={setSelectedChunk} />
            </div>
          )}
          {activePanel === 'ingest' && <DataIngestion />}
        </div>
      </div>

      <div className="flex-1 flex flex-col relative bg-background">
        <div className={`flex-1 flex flex-col h-full ${selectedChunk ? 'hidden' : ''}`}>
          {activePanel === 'chat' || activePanel === 'explorer' ? (
            <Chat onSelectChunk={setSelectedChunk} />
          ) : (
            <div className="flex-1 flex items-center justify-center text-textMuted">
              <div className="text-center">
                <Upload size={32} className="mx-auto mb-2 opacity-30" />
                <p className="text-sm">Switch to Chat or Explorer tab</p>
              </div>
            </div>
          )}
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
