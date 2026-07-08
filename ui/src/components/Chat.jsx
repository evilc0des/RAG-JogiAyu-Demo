import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import api from '../api';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, AlertCircle, Info, Brain } from 'lucide-react';

export default function Chat({ onSelectChunk }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Namaste! I am your Ayurvedic Medicine Educational Assistant. Ask me about symptoms, treatments, doshas, herbal remedies, or any Ayurvedic concepts from the indexed knowledge base.', chunks: [], grounded: true, abstained: false, reason: null, hop_trace: null }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [tooltip, setTooltip] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input.trim() };
    const chatHistory = messages.filter(m => !m.error).map(m => ({
      role: m.role,
      content: m.content
    }));

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.post(`/query`, {
        query: userMessage.content,
        chat_history: chatHistory,
        multi_hop: thinkingMode,
      });

      const { answer_text, citations, grounded, abstained, reason, chunks, hop_trace } = res.data;
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answer_text || "I don't have an answer for that.",
        citations,
        grounded,
        abstained,
        reason,
        chunks,
        hop_trace: hop_trace || null,
      }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.response?.data?.detail || err.message}`,
        error: true
      }]);
    }
    setLoading(false);
  };

  const showCitationTooltip = (e, chunk) => {
    const rect = e.target.getBoundingClientRect();
    const viewportWidth = window.innerWidth;
    let left = rect.left + rect.width / 2;
    const maxRight = left + 144;
    if (maxRight > viewportWidth - 8) {
      left = viewportWidth - 144 - 8;
    }
    if (left < 8) left = 8;
    setTooltip({
      text: chunk?.text || '',
      left,
      top: rect.top - 8,
    });
  };

  const hideCitationTooltip = () => setTooltip(null);

  return (
    <div className="flex h-full relative">
      {/* Chat Area */}
      <div className="flex flex-col flex-1 transition-all duration-300">
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                msg.role === 'user' ? 'bg-primary' : msg.error ? 'bg-red-500/20 text-red-500' : 'bg-surface border border-border'
              }`}>
                {msg.role === 'user' ? <User size={16} className="text-white" /> : <Bot size={16} />}
              </div>
              
              <div className={`max-w-[80%] rounded-2xl px-5 py-4 ${
                msg.role === 'user' 
                  ? 'bg-primary text-white rounded-tr-sm' 
                  : msg.error
                    ? 'bg-red-500/10 border border-red-500/30 text-red-400 rounded-tl-sm'
                    : 'glass rounded-tl-sm'
              }`}>
                {/* Meta info for assistant */}
                {msg.role === 'assistant' && !msg.error && (msg.abstained || !msg.grounded) && (
                  <div className="flex items-center gap-2 mb-3 text-xs px-2 py-1.5 rounded-md bg-red-500/10 text-red-400 border border-red-500/20">
                    <AlertCircle size={14} />
                    <span>
                      {msg.abstained ? 'Abstained from answering' : 'Potential hallucination detected'}
                      {msg.reason && `: ${msg.reason}`}
                    </span>
                  </div>
                )}

                <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert text-white' : 'prose-invert text-text'}`}>
                  <ReactMarkdown
                    components={{
                      p: ({node, ...props}) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
                      a: ({node, ...props}) => {
                        if (props.href?.startsWith('#citation-')) {
                          const citationId = props.href.replace('#citation-', '');
                          const citation = msg.citations?.find(c => c.citation_id === citationId);
                          const chunk = citation ? msg.chunks?.find(c => c.chunk_id === citation.section_id) : null;
                          const hasChunk = chunk?.text;

                          return (
                            <span className="inline-block ml-[2px]">
                              <sup
                                className={`font-bold px-[2px] transition-colors ${
                                  hasChunk
                                    ? 'text-primary hover:text-primaryHover cursor-pointer'
                                    : 'text-textMuted opacity-60'
                                }`}
                                onMouseEnter={hasChunk ? (e) => showCitationTooltip(e, chunk) : undefined}
                                onMouseLeave={hasChunk ? hideCitationTooltip : undefined}
                                onClick={hasChunk ? (e) => {
                                  e.preventDefault();
                                  e.stopPropagation();
                                  hideCitationTooltip();
                                  onSelectChunk(chunk);
                                } : undefined}
                                title={!hasChunk ? 'No chunk data available for this citation' : undefined}
                              >
                                {citationId}
                              </sup>
                            </span>
                          );
                        }
                        return <a className="text-primary hover:underline" {...props} />
                      }
                    }}
                  >
                    {msg.content.replace(/(?<!\[)\[(S\d+)\]/g, '[$1](#citation-$1)')}
                  </ReactMarkdown>
                </div>

                {/* Citations block */}
                {msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-border/50 flex flex-wrap gap-2">
                    {msg.citations.map((cit, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          const chk = msg.chunks?.find(c => c.chunk_id === cit.section_id);
                          if (chk) onSelectChunk(chk);
                        }}
                        onMouseEnter={(e) => {
                          const chk = msg.chunks?.find(c => c.chunk_id === cit.section_id);
                          showCitationTooltip(e, chk);
                        }}
                        onMouseLeave={hideCitationTooltip}
                        className="text-xs bg-surface border border-border px-2 py-1 rounded-md hover:bg-border transition-colors flex items-center gap-1.5"
                      >
                        <Info size={12} className="text-primary" />
                        {cit.citation_id}
                      </button>
                    ))}
                  </div>
                )}

                {/* Hop trace */}
                {msg.hop_trace && msg.hop_trace.length > 0 && (
                  <details className="mt-3 pt-3 border-t border-border/50">
                    <summary className="text-xs text-primary font-medium cursor-pointer hover:text-primaryHover transition-colors flex items-center gap-1.5">
                      <Brain size={12} />
                      Thinking trace ({msg.hop_trace.length} hop{msg.hop_trace.length !== 1 ? 's' : ''})
                    </summary>
                    <div className="mt-2 space-y-2">
                      {msg.hop_trace.map((hop, idx) => (
                        <div key={idx} className="bg-surface/50 border border-border rounded-lg p-3">
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-primary bg-primary/10 px-2 py-0.5 rounded">
                              Hop {hop.hop_number}
                            </span>
                            <span className={`text-[10px] px-2 py-0.5 rounded ${
                              hop.action === 'search'
                                ? 'bg-amber-500/10 text-amber-400'
                                : 'bg-green-500/10 text-green-400'
                            }`}>
                              {hop.action}
                            </span>
                          </div>
                          {hop.sub_queries && hop.sub_queries.length > 0 && (
                            <div className="space-y-1">
                              <span className="text-[10px] text-textMuted uppercase tracking-wider">Sub-queries</span>
                              {hop.sub_queries.map((sq, qi) => (
                                <div key={qi} className="text-xs text-text bg-background rounded px-2 py-1 font-mono">
                                  {sq}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-surface border border-border flex items-center justify-center">
                <Bot size={16} />
              </div>
              <div className="glass rounded-2xl rounded-tl-sm px-5 py-4 flex items-center gap-2">
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-background border-t border-border z-10">
          <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto flex items-end gap-2">
            <label
              className={`relative h-[52px] px-2 rounded-xl flex items-center gap-2 text-xs font-medium transition-all shrink-0 cursor-pointer select-none ${
                thinkingMode
                  ? 'bg-primary/20 text-primary border border-primary/40'
                  : 'bg-surface border border-border text-textMuted'
              }`}
              title={thinkingMode ? 'Thinking Mode ON — multi-hop retrieval enabled' : 'Thinking Mode OFF — single-pass retrieval'}
            >
              <Brain size={16} className={thinkingMode ? 'text-primary' : 'opacity-40'} />
              <span className="hidden sm:inline text-[11px]">Thinking Mode</span>
              <div className={`relative w-8 h-5 rounded-full transition-colors duration-200 ${
                thinkingMode ? 'bg-primary' : 'bg-border'
              }`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-all duration-200 ${
                  thinkingMode ? 'left-3.5' : 'left-0.5'
                }`} />
              </div>
              <input
                type="checkbox"
                checked={thinkingMode}
                onChange={() => setThinkingMode(!thinkingMode)}
                className="sr-only"
              />
            </label>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask a question..."
              className="w-full bg-surface border border-border rounded-xl px-4 py-3 min-h-[52px] max-h-32 focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 resize-none transition-shadow custom-scrollbar text-sm"
              rows={1}
            />
            <button 
              type="submit" 
              disabled={!input.trim() || loading}
              className="bg-primary hover:bg-primaryHover text-white h-[52px] w-[52px] rounded-xl flex items-center justify-center shrink-0 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-primary/20"
            >
              <Send size={20} />
            </button>
          </form>
          <div className="text-center mt-2 text-[10px] text-textMuted/60">
            AI can make mistakes. Verify information from citations.
          </div>
        </div>
      </div>

      {/* Portal tooltip — rendered to document.body to avoid overflow clipping */}
      {tooltip && createPortal(
        <div
          className="fixed z-[9999] w-72 p-3 bg-surface border border-border text-xs text-text rounded shadow-2xl leading-relaxed pointer-events-none"
          style={{
            left: `${tooltip.left}px`,
            top: `${tooltip.top}px`,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="line-clamp-5">{tooltip.text}</div>
        </div>,
        document.body
      )}
    </div>
  );
}
