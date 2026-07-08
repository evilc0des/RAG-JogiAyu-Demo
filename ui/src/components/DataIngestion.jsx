import React, { useState, useRef, useEffect, useCallback } from 'react';
import api from '../api';
import { Upload, FileText, Trash2, CheckCircle, XCircle, Loader2, Database, File, X, ArrowUp } from 'lucide-react';

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function FileBadge({ file, onRemove }) {
  return (
    <div className="flex items-center gap-2 bg-surface border border-border rounded-lg px-3 py-2 text-xs">
      <File size={14} className="text-primary shrink-0" />
      <span className="truncate flex-1">{file.name}</span>
      <span className="text-textMuted shrink-0">{formatSize(file.size)}</span>
      {onRemove && (
        <button onClick={() => onRemove(file)} className="text-textMuted hover:text-red-400 transition-colors shrink-0">
          <X size={14} />
        </button>
      )}
    </div>
  );
}

function ProgressBar({ progress, message, status }) {
  const pct = Math.round(progress * 100);
  const isError = status === 'failed';
  const isDone = status === 'completed';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className={`flex items-center gap-1.5 ${isDone ? 'text-green-400' : isError ? 'text-red-400' : 'text-text'}`}>
          {isDone ? <CheckCircle size={12} /> : isError ? <XCircle size={12} /> : <Loader2 size={12} className="animate-spin" />}
          {message}
        </span>
        <span className="text-textMuted tabular-nums">{pct}%</span>
      </div>
      <div className="w-full h-1.5 bg-surface rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${
            isError ? 'bg-red-500' : isDone ? 'bg-green-500' : 'bg-primary'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function DataIngestion() {
  const [tab, setTab] = useState('upload');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasteTitle, setPasteTitle] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const fileInputRef = useRef(null);

  const handleFilesSelected = (e) => {
    const files = Array.from(e.target.files || []);
    setSelectedFiles(prev => [...prev, ...files]);
    e.target.value = '';
  };

  const removeFile = (file) => {
    setSelectedFiles(prev => prev.filter(f => f !== file));
  };

  const pollJobStatus = useCallback(async (jobId) => {
    for (let i = 0; i < 120; i++) {
      try {
        const res = await api.get(`/api/ingest/status/${jobId}`);
        const job = res.data;
        setLoading(true);
        setStatus({
          type: job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'progress',
          jobId,
          progress: job.progress || 0,
          message: job.message || 'Processing...',
          details: job,
        });
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'partial') {
          setLoading(false);
          if (job.status === 'completed') setSelectedFiles([]);
          return;
        }
      } catch (e) {
        console.error('Poll error:', e);
      }
      await new Promise(r => setTimeout(r, 1000));
    }
    setLoading(false);
    setStatus({ type: 'error', message: 'Ingestion timed out' });
  }, []);

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) return;

    setLoading(true);
    setStatus(null);

    try {
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append('files', file);
      }
      const res = await api.post('/api/ingest/files', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      pollJobStatus(res.data.job_id);
    } catch (err) {
      setStatus({ type: 'error', message: err.response?.data?.detail || err.message });
      setLoading(false);
    }
  };

  const handlePasteText = async () => {
    if (!pasteText.trim()) return;
    setLoading(true);
    setStatus(null);

    try {
      const res = await api.post('/api/ingest/text', {
        text: pasteText,
        title: pasteTitle.trim() || 'Pasted Text',
      });
      setPasteText('');
      setPasteTitle('');
      pollJobStatus(res.data.job_id);
    } catch (err) {
      setStatus({ type: 'error', message: err.response?.data?.detail || err.message });
      setLoading(false);
    }
  };

  const handleDeleteAll = async () => {
    if (!confirm('Delete ALL documents and chunks? This cannot be undone.')) return;
    setLoading(true);
    try {
      const res = await api.delete('/api/documents');
      setStatus({ type: 'success', message: res.data.status });
    } catch (err) {
      setStatus({ type: 'error', message: err.response?.data?.detail || err.message });
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-border bg-surface/50">
        <h2 className="text-sm font-semibold flex items-center gap-1.5">
          <Database size={14} className="text-primary" />
          Data Ingestion
        </h2>
      </div>

      <div className="flex border-b border-border">
        {['upload', 'paste', 'manage'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-2 text-xs font-medium transition-colors ${
              tab === t
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-textMuted hover:text-text'
            }`}
          >
            {t === 'upload' && <Upload size={12} className="inline mr-1" />}
            {t === 'paste' && <FileText size={12} className="inline mr-1" />}
            {t === 'manage' && <Database size={12} className="inline mr-1" />}
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {tab === 'upload' && (
          <div className="space-y-3">
            <p className="text-xs text-textMuted">
              Upload files (.txt, .pdf, .json, .csv, .md) to ingest into the RAG system.
            </p>

            <div
              className="border-2 border-dashed border-border rounded-lg p-5 text-center hover:border-primary/50 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <ArrowUp size={22} className="mx-auto mb-1.5 text-textMuted" />
              <p className="text-xs text-textMuted">Click to select files</p>
              <p className="text-[10px] text-textMuted/60 mt-0.5">or drop files here</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.pdf,.json,.csv,.md,.log"
              onChange={handleFilesSelected}
              className="hidden"
            />

            {selectedFiles.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[10px] text-textMuted uppercase tracking-wider font-semibold">
                  Selected Files ({selectedFiles.length})
                </p>
                {selectedFiles.map((file, i) => (
                  <FileBadge key={`${file.name}-${i}`} file={file} onRemove={removeFile} />
                ))}
              </div>
            )}

            {status?.type === 'progress' && (
              <ProgressBar progress={status.progress} message={status.message} status="in_progress" />
            )}

            <button
              type="submit"
              onClick={handleFileUpload}
              disabled={loading || selectedFiles.length === 0}
              className="w-full bg-primary text-white py-2 rounded-lg text-sm font-medium hover:bg-primaryHover disabled:opacity-50 transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 size={14} className="animate-spin" /> Processing...
                </span>
              ) : 'Upload & Ingest'}
            </button>
          </div>
        )}

        {tab === 'paste' && (
          <div className="space-y-3">
            <p className="text-xs text-textMuted">
              Paste transcript or educational text directly.
            </p>
            <input
              type="text"
              value={pasteTitle}
              onChange={e => setPasteTitle(e.target.value)}
              placeholder="Title (optional)"
              className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-primary/50"
              disabled={loading}
            />
            <textarea
              value={pasteText}
              onChange={e => setPasteText(e.target.value)}
              placeholder="Paste text here..."
              rows={8}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary/50 resize-none custom-scrollbar"
              disabled={loading}
            />

            {status?.type === 'progress' && (
              <ProgressBar progress={status.progress} message={status.message} status="in_progress" />
            )}

            <button
              onClick={handlePasteText}
              disabled={loading || !pasteText.trim()}
              className="w-full bg-primary text-white py-2 rounded-lg text-sm font-medium hover:bg-primaryHover disabled:opacity-50 transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 size={14} className="animate-spin" /> Processing...
                </span>
              ) : 'Ingest Text'}
            </button>
          </div>
        )}

        {tab === 'manage' && (
          <div className="space-y-3">
            <p className="text-xs text-textMuted">
              Manage ingested documents. Deleting removes chunks from the index.
            </p>
            <button
              onClick={handleDeleteAll}
              disabled={loading}
              className="w-full bg-red-500/10 border border-red-500/30 text-red-400 py-2 rounded-lg text-sm font-medium hover:bg-red-500/20 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              <Trash2 size={14} />
              {loading ? 'Deleting...' : 'Delete All Documents'}
            </button>
          </div>
        )}

        {status && status.type !== 'progress' && (
          <div className={`p-3 rounded-lg border text-xs ${
            status.type === 'success'
              ? 'bg-green-500/10 border-green-500/30 text-green-400'
              : 'bg-red-500/10 border-red-500/30 text-red-400'
          }`}>
            <div className="flex items-center gap-1.5 font-medium mb-1">
              {status.type === 'success' ? <CheckCircle size={14} /> : <XCircle size={14} />}
              {status.type === 'success' ? 'Success' : 'Error'}
            </div>
            <p>{status.message}</p>
            {status.details?.errors?.length > 0 && (
              <ul className="mt-1 list-disc list-inside">
                {status.details.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
