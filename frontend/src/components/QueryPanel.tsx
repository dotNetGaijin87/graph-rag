import { useState, type FormEvent } from 'react';
import { Alert, Box, Button, CircularProgress, Paper, TextField, Typography } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { api, ApiError } from '../api/client';
import type { AnswerResponse } from '../api/types';
import { AnswerView } from './AnswerView';

export function QueryPanel() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);

  const canSubmit = question.trim().length > 0 && !loading;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    try {
      const response = await api.ask(question.trim());
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <Paper variant="outlined" sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        2 · Ask a question
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Your question is embedded, matched against stored chunks, expanded over the graph,
        and answered by the local LLM — grounded only in what you added.
      </Typography>

      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <TextField
          label="Ask something about the text you added…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          multiline
          minRows={3}
          fullWidth
        />
        <Button
          type="submit"
          variant="contained"
          disabled={!canSubmit}
          startIcon={loading ? <CircularProgress size={18} color="inherit" /> : <SendIcon />}
        >
          {loading ? 'Thinking…' : 'Ask'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      {result && (
        <Box sx={{ mt: 2 }}>
          <AnswerView result={result} />
        </Box>
      )}
    </Paper>
  );
}
