import React, { useState, useEffect } from 'react';
import api from '../api';
import { X, FileText, Database, Type, ListTree } from 'lucide-react';

export default function ChunkViewer({ chunk, onClose }) {
  const [pageChunk, setPageChunk] = useState(null);
  const [loadingPage, setLoadingPage] = useState(false);
  const [highlightText, setHighlightText] = useState(null);
  const [highlightChunkId, setHighlightChunkId] = useState(null);

  useEffect(() => {
    if (!chunk) return;
    if (chunk.chunk_type === 'page') {
      setPageChunk(chunk);
    } else if (chunk.doc_id) {
      setLoadingPage(true);
      api.get(`/api/pages/${chunk.doc_id}`)
        .then(res => {
          setPageChunk(res.data.page);
          setLoadingPage(false);
        })
        .catch(err => {
          console.error("Failed to load parent page", err);
          setPageChunk(chunk);
          setLoadingPage(false);
        });
    } else {
      setPageChunk(chunk);
    }
  }, [chunk]);

  useEffect(() => {
    if (!chunk?.child_ids?.length) {
      setHighlightText(null);
      setHighlightChunkId(null);
      return;
    }
    const childId = chunk.child_ids[0];
    setHighlightChunkId(childId);
    api.get(`/api/chunks/${childId}`)
      .then(res => {
        setHighlightText(res.data.chunk.text);
      })
      .catch(err => {
        console.error("Failed to load child chunk for highlighting", err);
        setHighlightText(null);
      });
  }, [chunk]);

  useEffect(() => {
    if (pageChunk && !loadingPage && highlightText) {
      const timer = setTimeout(() => {
        const el = document.getElementById('highlighted-chunk');
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [pageChunk, loadingPage, highlightText]);

  if (!chunk) return null;

  const deduplicateCumulativeText = (text) => {
    const lines = text.split('\n');
    const n = lines.length;
    const keep = new Array(n).fill(true);

    for (let i = 0; i < n; i++) {
      for (let blockLen = 1; i + blockLen * 2 <= n; blockLen++) {
        let match = true;
        for (let j = 0; j < blockLen; j++) {
          if (lines[i + j] !== lines[i + blockLen + j]) {
            match = false;
            break;
          }
        }
        if (match) {
          if (i + blockLen * 2 < n) {
            for (let j = 0; j < blockLen; j++) {
              keep[i + j] = false;
            }
          }
          break;
        }
      }
    }

    return lines.filter((_, idx) => keep[idx]).join('\n');
  };

  const normalizeText = (text) => text.replace(/\s+/g, ' ').trim();

  const findAndHighlight = (sourceText, searchText, displayText) => {
    const parts = sourceText.split(searchText);
    if (parts.length === 1) return null;
    return (
      <>
        {parts.map((part, i) => (
          <React.Fragment key={i}>
            {part}
            {i < parts.length - 1 && (
              <mark className="bg-primary/40 text-text rounded px-1 font-semibold border border-primary/50 shadow-[0_0_10px_rgba(59,130,246,0.3)] inline-block" id="highlighted-chunk">
                {displayText}
              </mark>
            )}
          </React.Fragment>
        ))}
      </>
    );
  };

  const renderHighlightedText = () => {
    const rawText = pageChunk?.text || "";
    const dedupText = deduplicateCumulativeText(rawText);
    const targetText = highlightText || chunk.text;

    if (!pageChunk || chunk.chunk_id === pageChunk.chunk_id || !targetText) {
      return dedupText;
    }

    const highlight = findAndHighlight(dedupText, targetText, targetText)
      || findAndHighlight(normalizeText(dedupText), normalizeText(targetText), targetText)
      || findAndHighlight(rawText, targetText, targetText)
      || findAndHighlight(normalizeText(rawText), normalizeText(targetText), targetText);

    if (highlight) return highlight;

    return dedupText;
  };

  return (
    <div className="flex flex-col h-full bg-background relative animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border bg-surface">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            {chunk.chunk_type === 'page' ? <FileText size={20} className="text-primary" /> : <Type size={20} className="text-primary" />}
          </div>
          <div>
            <h2 className="text-lg font-semibold text-text">{pageChunk?.title || chunk.title || chunk.chunk_id}</h2>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xs bg-border px-2 py-0.5 rounded text-textMuted uppercase tracking-wider font-semibold">
                {chunk.chunk_type === 'page' ? 'Page' : 'Page Context'}
              </span>
              {chunk.doc_id && (
                <span className="text-xs text-textMuted flex items-center gap-1">
                  <Database size={12} /> {chunk.doc_id}
                </span>
              )}
            </div>
          </div>
        </div>
        <button 
          onClick={onClose}
          className="p-2 rounded-full hover:bg-border text-textMuted hover:text-text transition-colors"
          title="Close Viewer"
        >
          <X size={20} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="glass p-6 rounded-xl prose prose-invert prose-sm max-w-none relative">
            {loadingPage ? (
              <div className="flex items-center justify-center p-8 text-textMuted">
                <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin mr-3"></div>
                Loading full page context...
              </div>
            ) : (
              <div className="whitespace-pre-wrap leading-relaxed">
                {renderHighlightedText()}
              </div>
            )}
          </div>
          
          {/* Metadata Table */}
          <div className="glass p-6 rounded-xl">
            <h3 className="text-sm font-semibold text-textMuted uppercase tracking-wider mb-4 flex items-center gap-2">
              <ListTree size={16} /> Chunk Metadata
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-textMuted mb-1">Target Chunk ID</div>
                <div className="text-sm font-mono bg-background/50 p-2 rounded border border-border">{highlightChunkId || chunk.chunk_id}</div>
              </div>
              {chunk.parent_id && (
                <div>
                  <div className="text-xs text-textMuted mb-1">Parent ID</div>
                  <div className="text-sm font-mono bg-background/50 p-2 rounded border border-border">{chunk.parent_id}</div>
                </div>
              )}
              {highlightChunkId && highlightChunkId !== chunk.chunk_id && (
                <div>
                  <div className="text-xs text-textMuted mb-1">Section ID</div>
                  <div className="text-sm font-mono bg-background/50 p-2 rounded border border-border">{chunk.chunk_id}</div>
                </div>
              )}
              {chunk.source_url && (
                <div className="md:col-span-2">
                  <div className="text-xs text-textMuted mb-1">Source URL</div>
                  <div className="text-sm text-primary hover:underline bg-background/50 p-2 rounded border border-border truncate">
                    <a href={chunk.source_url} target="_blank" rel="noopener noreferrer">{chunk.source_url}</a>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
