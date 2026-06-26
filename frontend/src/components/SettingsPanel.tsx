import { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  FormControlLabel,
  Paper,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import { api, ApiError } from '../api/client';
import type { EditableSettings, Settings } from '../api/types';

type NumberKey = 'chunk_size' | 'chunk_overlap' | 'top_k' | 'max_extraction_chars';

interface NumberField {
  key: NumberKey;
  label: string;
  help: string;
  min: number;
  max: number;
}

const NUMBER_FIELDS: NumberField[] = [
  {
    key: 'chunk_size',
    label: 'Chunk size (characters)',
    help: 'Length of each text chunk before embedding. Affects newly added text.',
    min: 50,
    max: 20000,
  },
  {
    key: 'chunk_overlap',
    label: 'Chunk overlap (characters)',
    help: 'Overlap between consecutive chunks. Must be smaller than chunk size.',
    min: 0,
    max: 19999,
  },
  {
    key: 'top_k',
    label: 'Top-K retrieved chunks',
    help: 'How many chunks to retrieve per question.',
    min: 1,
    max: 50,
  },
  {
    key: 'max_extraction_chars',
    label: 'Max extraction characters',
    help: 'Cap on how much text is sent to the LLM for entity/relationship extraction.',
    min: 200,
    max: 200000,
  },
];

export function SettingsPanel() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [form, setForm] = useState<EditableSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  function applySettings(s: Settings) {
    setSettings(s);
    setForm({
      chunk_size: s.chunk_size,
      chunk_overlap: s.chunk_overlap,
      top_k: s.top_k,
      enable_entity_extraction: s.enable_entity_extraction,
      max_extraction_chars: s.max_extraction_chars,
    });
  }

  useEffect(() => {
    let active = true;
    api
      .getSettings()
      .then((s) => {
        if (active) applySettings(s);
      })
      .catch((err) => {
        if (active) setError(err instanceof ApiError ? err.message : 'Failed to load settings.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  function setNumber(key: NumberKey, value: string) {
    setSaved(false);
    setForm((f) => (f ? { ...f, [key]: Number(value) } : f));
  }

  async function handleSave() {
    if (!form) return;
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const updated = await api.updateSettings(form);
      applySettings(updated);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to save settings.');
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!form || !settings) {
    return <Alert severity="error">{error ?? 'Settings are unavailable.'}</Alert>;
  }

  return (
    <Paper variant="outlined" sx={{ p: 3, maxWidth: 640 }}>
      <Typography variant="h6" gutterBottom>
        RAG settings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Tune the retrieval pipeline. Changes apply immediately and are kept in memory until
        the backend restarts. Chunking changes affect newly added text only.
      </Typography>

      <Stack spacing={3}>
        {NUMBER_FIELDS.map((field) => (
          <TextField
            key={field.key}
            type="number"
            label={field.label}
            value={form[field.key]}
            onChange={(e) => setNumber(field.key, e.target.value)}
            helperText={field.help}
            inputProps={{ min: field.min, max: field.max }}
            fullWidth
            size="small"
          />
        ))}

        <FormControlLabel
          control={
            <Switch
              checked={form.enable_entity_extraction}
              onChange={(e) => {
                setSaved(false);
                setForm((f) => (f ? { ...f, enable_entity_extraction: e.target.checked } : f));
              }}
            />
          }
          label="Entity & relationship extraction (GraphRAG)"
        />

        <Box>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={saving}
            startIcon={saving ? <CircularProgress size={18} color="inherit" /> : <SaveIcon />}
          >
            {saving ? 'Saving…' : 'Save settings'}
          </Button>
        </Box>

        {error && <Alert severity="error">{error}</Alert>}
        {saved && <Alert severity="success">Settings saved.</Alert>}

        <Divider />

        <Box>
          <Typography variant="overline" color="text.secondary">
            Models (fixed at startup)
          </Typography>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="LLM model" value={settings.llm_model} size="small" fullWidth disabled />
            <TextField
              label="Embedding model"
              value={settings.embedding_model}
              size="small"
              fullWidth
              disabled
              helperText={`Vector dimension: ${settings.embedding_dim}. Changing the embedding model would invalidate the existing index, so it is configured via environment only.`}
            />
          </Stack>
        </Box>
      </Stack>
    </Paper>
  );
}
