import { createTheme } from '@mui/material/styles';

// Dark theme that keeps the original GraphRAG look-and-feel.
export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#6c7bff' },
    secondary: { main: '#2ecc8f' },
    background: { default: '#0f1220', paper: '#1a1f35' },
    error: { main: '#ff6b6b' },
    success: { main: '#2ecc8f' },
    text: { primary: '#e7e9f3', secondary: '#9aa3c7' },
    divider: '#2d3556',
  },
  shape: { borderRadius: 10 },
  typography: {
    fontFamily: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  },
  components: {
    MuiPaper: { styleOverrides: { root: { backgroundImage: 'none' } } },
    MuiButton: { defaultProps: { disableElevation: true } },
  },
});
