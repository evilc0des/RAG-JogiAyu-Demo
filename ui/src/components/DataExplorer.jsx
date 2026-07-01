import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { ChevronRight, ChevronDown, FileText, Type } from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';

const API_BASE = 'http://localhost:8000/api';

function ChunkNode({ chunk, onSelect, level = 0, onResize }) {
  const [expanded, setExpanded] = useState(false);
  const [children, setChildren] = useState(null);
  const [loading, setLoading] = useState(false);

  const hasChildren = chunk.children_ids && chunk.children_ids.length > 0;

  const toggleExpand = async () => {
    if (!expanded && hasChildren && !children) {
      setLoading(true);
      try {
        const res = await axios.get(`${API_BASE}/chunks/${chunk.chunk_id}`);
        setChildren(res.data.children);
      } catch (err) {
        console.error("Failed to load children", err);
      }
      setLoading(false);
    }
    setExpanded(!expanded);
  };

  const getIcon = () => {
    if (chunk.chunk_type === 'page') return <FileText size={16} className="text-primary" />;
    return <Type size={16} className="text-textMuted" />;
  };

  return (
    <div className="select-none">
      <div 
        className={`flex items-center py-1.5 px-2 rounded-md hover:bg-surface cursor-pointer transition-colors ${level === 0 ? 'mt-1' : ''}`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
      >
        <div 
          className="w-5 h-5 flex items-center justify-center mr-1"
          onClick={(e) => { e.stopPropagation(); toggleExpand(); }}
        >
          {loading ? (
            <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          ) : hasChildren ? (
            expanded ? <ChevronDown size={16} className="text-textMuted" /> : <ChevronRight size={16} className="text-textMuted" />
          ) : null}
        </div>
        
        <div 
          className="flex items-center flex-1 overflow-hidden"
          onClick={() => onSelect(chunk)}
        >
          <span className="mr-2">{getIcon()}</span>
          <span className="truncate text-sm font-medium">
            {chunk.title || chunk.chunk_id.substring(0, 8)}
          </span>
          <span className="ml-2 text-xs text-textMuted opacity-60 bg-border px-1.5 rounded shrink-0">
            {chunk.chunk_type}
          </span>
        </div>
      </div>
      
      {expanded && children && (
        <div className="border-l border-border ml-[22px]">
          {children.map(child => (
            <ChunkNode 
              key={child.chunk_id} 
              chunk={child} 
              onSelect={onSelect} 
              level={level + 1} 
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function DataExplorer({ onSelectChunk }) {
  const [chunksMap, setChunksMap] = useState({});
  const [total, setTotal] = useState(0);
  const [fetchingRanges, setFetchingRanges] = useState(new Set());
  const [initialLoading, setInitialLoading] = useState(true);
  
  const parentRef = useRef(null);
  
  const fetchPage = useCallback(async (pageIdx, limit) => {
    setFetchingRanges(prev => new Set(prev).add(pageIdx));
    const offset = pageIdx * limit;
    try {
      const res = await axios.get(`${API_BASE}/chunks?limit=${limit}&offset=${offset}`);
      setTotal(res.data.total);
      setChunksMap(prev => {
        const next = { ...prev };
        res.data.chunks.forEach((chunk, i) => {
          next[offset + i] = chunk;
        });
        return next;
      });
    } catch(e) {
      console.error(e);
      setFetchingRanges(prev => {
        const next = new Set(prev);
        next.delete(pageIdx);
        return next;
      });
    } finally {
      if (pageIdx === 0) setInitialLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchPage(0, 50);
  }, [fetchPage]);

  const rowVirtualizer = useVirtualizer({
    count: total || 0,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 36,
    overscan: 15,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();

  useEffect(() => {
    if (!virtualItems.length) return;
    
    const limit = 50;
    virtualItems.forEach(item => {
      const pageIdx = Math.floor(item.index / limit);
      if (!fetchingRanges.has(pageIdx) && !chunksMap[pageIdx * limit]) {
        fetchPage(pageIdx, limit);
      }
    });
  }, [virtualItems, fetchingRanges, chunksMap, fetchPage]);

  if (initialLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-40 text-textMuted gap-3">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
        <p className="text-sm">Loading chunks...</p>
      </div>
    );
  }

  if (total === 0) {
    return (
      <div className="text-center p-8 text-textMuted border border-dashed border-border rounded-lg mt-4 mx-2">
        <p className="text-sm">No chunks found in database.</p>
        <p className="text-xs mt-1">Run indexing first.</p>
      </div>
    );
  }

  return (
    <div 
      ref={parentRef} 
      className="h-full overflow-y-auto custom-scrollbar overflow-x-hidden"
    >
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualItems.map((virtualRow) => {
          const chunk = chunksMap[virtualRow.index];
          
          return (
            <div
              key={virtualRow.index}
              data-index={virtualRow.index}
              ref={rowVirtualizer.measureElement}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {chunk ? (
                <ChunkNode 
                  chunk={chunk} 
                  onSelect={onSelectChunk}
                />
              ) : (
                <div className="flex items-center py-2 px-3">
                  <div className="h-4 bg-surface rounded w-full animate-pulse opacity-50"></div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
