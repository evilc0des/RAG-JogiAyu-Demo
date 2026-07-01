import React, { useState, useRef, useEffect } from 'react';
import api from '../api';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, AlertCircle, Info } from 'lucide-react';

export default function Chat({ onSelectChunk }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your RAG Assistant. Ask me anything about the indexed documents.', sections: [] }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
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
        chat_history: chatHistory
      });

      const { answer_text, citations, grounded, abstained, reason, sections } = res.data;
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answer_text || "I don't have an answer for that.",
        citations,
        grounded,
        abstained,
        reason,
        sections
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
                          const section = citation ? msg.sections?.find(s => s.chunk_id === citation.section_id) : null;
                          
                          return (
                            <span className="relative inline-block group cursor-pointer ml-[2px]">
                              <sup 
                                className="text-primary font-bold hover:text-primaryHover transition-colors px-[2px]"
                                onClick={(e) => {
                                  e.preventDefault();
                                  if (section) onSelectChunk(section);
                                }}
                              >
                                {citationId}
                              </sup>
                              {section && (
                                <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 w-64 p-3 bg-surface border border-border text-xs text-text rounded shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 pointer-events-none line-clamp-5 leading-relaxed">
                                  {section.text}
                                </span>
                              )}
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
                          const sec = msg.sections?.find(s => s.chunk_id === cit.section_id);
                          if (sec) onSelectChunk(sec);
                        }}
                        className="text-xs bg-surface border border-border px-2 py-1 rounded-md hover:bg-border transition-colors flex items-center gap-1.5"
                      >
                        <Info size={12} className="text-primary" />
                        {cit.citation_id}
                      </button>
                    ))}
                  </div>
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
    </div>
  );
}
