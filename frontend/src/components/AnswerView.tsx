import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import type { AnswerResponse } from '../api/types';

interface Props {
  result: AnswerResponse;
}

export function AnswerView({ result }: Props) {
  const { answer, context } = result;

  return (
    <Box>
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        Answer
      </Typography>
      <Typography sx={{ whiteSpace: 'pre-wrap', mb: 2 }}>{answer}</Typography>

      <Accordion variant="outlined" disableGutters sx={{ '&::before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2" color="text.secondary">
            Sources — {context.chunks.length} passages, {context.facts.length} graph facts
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          {context.facts.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="overline" color="text.secondary">
                Knowledge-graph facts
              </Typography>
              <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                {context.facts.map((fact, i) => (
                  <Typography key={i} variant="body2">
                    <strong>{fact.source}</strong>{' '}
                    <Box component="span" sx={{ color: 'text.secondary' }}>
                      {fact.type.replace(/_/g, ' ').toLowerCase()}
                    </Box>{' '}
                    <strong>{fact.target}</strong>
                  </Typography>
                ))}
              </Stack>
            </Box>
          )}

          {context.chunks.length > 0 && (
            <Box>
              <Typography variant="overline" color="text.secondary">
                Retrieved passages
              </Typography>
              <Stack spacing={1} sx={{ mt: 0.5 }}>
                {context.chunks.map((chunk) => (
                  <Paper
                    key={chunk.chunk_id}
                    variant="outlined"
                    sx={{ p: 1.5, bgcolor: 'background.default' }}
                  >
                    <Stack
                      direction="row"
                      spacing={1}
                      alignItems="center"
                      sx={{ flexWrap: 'wrap', mb: 0.5 }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        score {chunk.score.toFixed(3)}
                      </Typography>
                      {chunk.entities.map((entity) => (
                        <Chip key={entity} label={entity} size="small" />
                      ))}
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {chunk.text}
                    </Typography>
                  </Paper>
                ))}
              </Stack>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
