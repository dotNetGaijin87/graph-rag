import { Box, Button, Paper, Stack, Typography } from '@mui/material';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import type { Stats } from '../api/types';

interface Props {
  stats: Stats | null;
  onReset: () => void;
}

export function StatsBar({ stats, onReset }: Props) {
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: 2,
      }}
    >
      <Stack direction="row" spacing={4}>
        <Stat label="Documents" value={stats?.documents} />
        <Stat label="Chunks" value={stats?.chunks} />
        <Stat label="Entities" value={stats?.entities} />
        <Stat label="Relationships" value={stats?.relationships} />
      </Stack>
      <Button color="error" size="small" startIcon={<DeleteOutlineIcon />} onClick={onReset}>
        Reset
      </Button>
    </Paper>
  );
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <Box>
      <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.1 }}>
        {value ?? '—'}
      </Typography>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}
      >
        {label}
      </Typography>
    </Box>
  );
}
