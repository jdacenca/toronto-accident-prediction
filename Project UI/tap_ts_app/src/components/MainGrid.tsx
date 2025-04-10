import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSelector } from "react-redux";
import { RootState } from "../redux/store";
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Copyright from '../internals/components/Copyright';
import BasicMap from './BasicMap';
import CustomizedDataGrid from './CustomizedDataGrid';
import PageViewsBarChart from './PageViewsBarChart';
import StatCard, { StatCardProps } from './StatCard';
import YearSelection from './YearSelection';

export default function MainGrid() {
  const year = useSelector((state: RootState) => state.tapApp.year);
  const [statData, setStatData] = useState<StatCardProps[]>([
    {
      title: 'Total Fatalities',
      value: '325',
      interval: 'Based on Year (' + year + ')',
      trend: 'down',
      data: [
        1640, 1250, 970, 1130, 1050, 900, 720, 1080, 900, 450, 920,
      ],
    },
    {
      title: 'Total Non-Fatalities',
      value: '200k',
      interval: 'Based on Year (' + year + ')',
      trend: 'neutral',
      data: [
        500, 400, 510, 530, 520, 600, 530, 520, 510, 730, 520, 510,
      ],
    },
  ]);

  const apiUrl = 'http://127.0.0.1:5000';

  useEffect(() => {
    const fetchData = async () => {

      // Fetch the total accidents for the selected year for Fatalities
      try {
        const response = await fetch(apiUrl + '/data/accident/total/' + year + '/Fatal', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: any = await response.json();

        const newData1 = [...statData];
        newData1[0].value = data.total.toString();
        newData1[0].data = data.per_month;
        newData1[0].interval = 'Based on Year (' + year + ')',
          setStatData(newData1);

      } catch (err) {
        console.error('Error fetching data:', err);
        console.error('Failed to load options. Please try again later.');
      }

      // Fetch the total accidents for the selected year for Non Fatalities
      try {
        const response = await fetch(apiUrl + '/data/accident/total/' + year + '/Non-Fatal%20Injury', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: any = await response.json();

        const newData2 = [...statData];
        newData2[1].value = data.total.toString();
        newData2[1].data = data.per_month;
        newData2[1].interval = 'Based on Year (' + year + ')',
          setStatData(newData2);

      } catch (err) {
        console.error('Error fetching data:', err);
        console.error('Failed to load options. Please try again later.');
      }
    };

    fetchData();
  }, [year]);
  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px', lg: '2400px' } }}>
      {/* cards */}
      <Stack direction="row" sx={{ gap: 1 }}>
        <Typography component="h4" variant="h4" sx={{ mb: 2 }}>
          Year:
        </Typography>
        <YearSelection />
      </Stack>
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        {statData.map((card, index) => (
          <Grid key={index} size={{ xs: 12, sm: 12, lg: 6 }}>
            <StatCard {...card} />
          </Grid>
        ))}
        <Grid size={{ xs: 12, sm: 12, lg: 12 }}>
          <PageViewsBarChart />
        </Grid>
      </Grid>
      <Grid size={{ xs: 4, md: 4 }}>
        <BasicMap />
      </Grid>

      <Typography component="h2" variant="h6" sx={{ mb: 2 }}>
        Details
      </Typography>
      <Grid container spacing={2} columns={12}>
        <Grid size={{ xs: 12, lg: 12 }}>
          <CustomizedDataGrid />
        </Grid>
      </Grid>
      <Copyright sx={{ my: 4 }} />
    </Box>
  );
}
