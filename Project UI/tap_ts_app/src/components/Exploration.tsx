import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSelector } from "react-redux";
import { RootState } from "../redux/store";
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Copyright from '../internals/components/Copyright';
import FieldCountChart from './FieldCountChart';
import MissingFieldCountChart from './MissingFieldCountChart';

export default function Exploration() {
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px', lg: '2400px'} }}>
      {/* cards */}
      <Stack direction="row" sx={{ gap: 1 }}>
        <Typography component="h4" variant="h4" sx={{ mb: 2 }}>
          Exploration 
        </Typography>
      </Stack>
      <Grid
        container
        spacing={2}
        columns={4}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        <Grid size={{ xs: 4, md: 4}}>
          <FieldCountChart />
        </Grid>
        <Grid size={{ xs: 4, md: 4}}>
          <MissingFieldCountChart />
        </Grid>
        <Grid size={{ xs: 4, md: 4}}>
        </Grid>
      </Grid>
      <Copyright sx={{ my: 4 }} />
    </Box>
  );
}
