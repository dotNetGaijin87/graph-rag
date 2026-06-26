import { useState, type FormEvent } from 'react';
import { Alert, Box, Button, CircularProgress, Paper, TextField, Typography } from '@mui/material';
import { api, ApiError } from '../api/client';
import type { IngestionReport } from '../api/types';

interface Props {
  onIngested: () => void;
}

export function IngestPanel({ onIngested }: Props) {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<IngestionReport | null>(null);

  const canSubmit = text.trim().length > 0 && !loading;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await api.ingest(text.trim(), title.trim());
      setReport(result);
      setText('');
      setTitle('');
      onIngested();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <Paper variant="outlined" sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        1 · Add knowledge
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Paste any text. It is chunked, embedded, and an entity/relationship graph is
        extracted and stored in Neo4j.
      </Typography>

      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <TextField
          label="Title (optional)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          disabled={loading}
          size="small"
          fullWidth
        />
        <TextField
          label="Paste text to remember…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          multiline
          minRows={8}
          fullWidth
        />
        <Button
          type="submit"
          variant="contained"
          disabled={!canSubmit}
          startIcon={loading ? <CircularProgress size={18} color="inherit" /> : undefined}
        >
          {loading ? 'Ingesting…' : 'Add to knowledge base'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      {report && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Stored “{report.title}” — {report.chunk_count} chunks, {report.entity_count}{' '}
          entities, {report.relationship_count} relationships.
        </Alert>
      )}
    </Paper>
  );
}
