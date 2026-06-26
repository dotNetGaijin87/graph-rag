import { lazy, Suspense, useCallback, useEffect, useState, type ReactNode } from 'react';
import {
  AppBar,
  Box,
  CircularProgress,
  Container,
  Stack,
  Tab,
  Tabs,
  Toolbar,
  Typography,
} from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import HubIcon from '@mui/icons-material/Hub';
import SettingsIcon from '@mui/icons-material/Settings';
import { api } from './api/client';
import type { Stats } from './api/types';
import { IngestPanel } from './components/IngestPanel';
import { QueryPanel } from './components/QueryPanel';
import { StatsBar } from './components/StatsBar';
import { SettingsPanel } from './components/SettingsPanel';

// Lazy-loaded so the heavy Cytoscape bundle is only fetched when the graph tab opens.
const GraphView = lazy(() =>
  import('./components/GraphView').then((m) => ({ default: m.GraphView })),
);

export default function App() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [tab, setTab] = useState(0);

  const refreshStats = useCallback(() => {
    api
      .stats()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  async function handleReset() {
    if (!window.confirm('Delete all stored knowledge?')) return;
    await api.reset().catch(() => undefined);
    refreshStats();
  }

  return (
    <Box sx={{ minHeight: '100vh', pb: 8 }}>
      <AppBar
        position="static"
        color="transparent"
        elevation={0}
        sx={{ borderBottom: 1, borderColor: 'divider' }}
      >
        <Toolbar sx={{ flexDirection: 'column', alignItems: 'flex-start', py: 1.5 }}>
          <Typography
            variant="h5"
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(90deg, #8590ff, #2ecc8f)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            GraphRAG Knowledge Base
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Local Retrieval-Augmented Generation · Neo4j · Ollama — fully offline
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ pt: 3 }}>
        <StatsBar stats={stats} onReset={handleReset} />

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mt: 3 }}>
          <Tabs value={tab} onChange={(_, value: number) => setTab(value)}>
            <Tab icon={<StorageIcon />} iconPosition="start" label="Knowledge base" />
            <Tab icon={<HubIcon />} iconPosition="start" label="Knowledge graph" />
            <Tab icon={<SettingsIcon />} iconPosition="start" label="Settings" />
          </Tabs>
        </Box>

        <TabPanel value={tab} index={0}>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="stretch">
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <IngestPanel onIngested={refreshStats} />
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <QueryPanel />
            </Box>
          </Stack>
        </TabPanel>

        <TabPanel value={tab} index={1}>
          <Suspense
            fallback={
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress />
              </Box>
            }
          >
            <GraphView />
          </Suspense>
        </TabPanel>

        <TabPanel value={tab} index={2}>
          <SettingsPanel />
        </TabPanel>
      </Container>
    </Box>
  );
}

function TabPanel({
  value,
  index,
  children,
}: {
  value: number;
  index: number;
  children: ReactNode;
}) {
  if (value !== index) return null;
  return <Box sx={{ pt: 2.5 }}>{children}</Box>;
}
