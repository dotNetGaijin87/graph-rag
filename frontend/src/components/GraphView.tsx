import { useCallback, useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { api, ApiError } from '../api/client';
import type { GraphData } from '../api/types';

const TYPE_COLORS: Record<string, string> = {
  Person: '#6c7bff',
  Organization: '#2ecc8f',
  Location: '#ff9f43',
  Concept: '#a66bff',
  Product: '#ff6b6b',
  Event: '#4bc0c0',
  Other: '#9aa3c7',
};

function colorForType(type: string): string {
  return TYPE_COLORS[type] ?? TYPE_COLORS.Other;
}

// Colour and size are driven by per-element `data` set when building elements below.
const CY_STYLE: cytoscape.StylesheetJson = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)',
      label: 'data(label)',
      color: '#e7e9f3',
      'font-size': 10,
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '90px',
      'text-outline-color': '#0f1220',
      'text-outline-width': 2,
      width: 'data(size)',
      height: 'data(size)',
    },
  },
  {
    selector: 'edge',
    style: {
      label: 'data(label)',
      width: 1.4,
      'line-color': '#3a4472',
      'target-arrow-color': '#3a4472',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      'font-size': 8,
      color: '#9aa3c7',
      'text-rotation': 'autorotate',
      'text-background-color': '#0f1220',
      'text-background-opacity': 1,
      'text-background-padding': '2px',
    },
  },
];

export function GraphView() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [counts, setCounts] = useState({ nodes: 0, edges: 0 });

  const renderGraph = useCallback((data: GraphData) => {
    const container = containerRef.current;
    if (!container) return;

    const nodeIds = new Set(data.nodes.map((n) => n.id));
    const validEdges = data.edges.filter(
      (e) => nodeIds.has(e.source) && nodeIds.has(e.target),
    );

    // Node size grows with its number of connections.
    const degree = new Map<string, number>();
    for (const e of validEdges) {
      degree.set(e.source, (degree.get(e.source) ?? 0) + 1);
      degree.set(e.target, (degree.get(e.target) ?? 0) + 1);
    }

    const elements: cytoscape.ElementDefinition[] = [
      ...data.nodes.map((n) => ({
        data: {
          id: n.id,
          label: n.id,
          color: colorForType(n.type),
          size: 22 + Math.min(degree.get(n.id) ?? 0, 8) * 5,
        },
      })),
      ...validEdges.map((e, i) => ({
        data: {
          id: `edge-${i}`,
          source: e.source,
          target: e.target,
          label: e.type.replace(/_/g, ' ').toLowerCase(),
        },
      })),
    ];

    cyRef.current?.destroy();
    cyRef.current = cytoscape({
      container,
      elements,
      style: CY_STYLE,
      layout: { name: 'cose', animate: false, padding: 30 },
      minZoom: 0.2,
      maxZoom: 2.5,
    });
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.graph();
      setCounts({ nodes: data.nodes.length, edges: data.edges.length });
      renderGraph(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load the graph.');
    } finally {
      setLoading(false);
    }
  }, [renderGraph]);

  useEffect(() => {
    load();
  }, [load]);

  // Clean up the cytoscape instance on unmount.
  useEffect(() => () => cyRef.current?.destroy(), []);

  return (
    <Paper variant="outlined" sx={{ p: 3 }}>
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="flex-start"
        spacing={2}
        sx={{ mb: 1.5 }}
      >
        <Box>
          <Typography variant="h6">Knowledge graph</Typography>
          <Typography variant="body2" color="text.secondary">
            Entities and the relationships between them, extracted from everything you've
            added. Drag nodes to explore, scroll to zoom.
          </Typography>
        </Box>
        <Button
          size="small"
          onClick={load}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
        >
          Refresh
        </Button>
      </Stack>

      <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', gap: 1.5, mb: 1.5 }}>
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <Stack key={type} direction="row" spacing={0.5} alignItems="center">
            <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: color }} />
            <Typography variant="caption" color="text.secondary">
              {type}
            </Typography>
          </Stack>
        ))}
      </Stack>

      <Box sx={{ position: 'relative' }}>
        <Box
          ref={containerRef}
          sx={{
            width: '100%',
            height: 560,
            bgcolor: 'background.default',
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
          }}
        />
        {!loading && counts.nodes === 0 && !error && (
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              p: 4,
              pointerEvents: 'none',
            }}
          >
            <Typography variant="body2" color="text.secondary">
              No entities yet. Add some text mentioning people, places, or concepts — with
              entity extraction enabled — and they'll appear here.
            </Typography>
          </Box>
        )}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      {counts.nodes > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          {counts.nodes} entities · {counts.edges} relationships
        </Typography>
      )}
    </Paper>
  );
}
