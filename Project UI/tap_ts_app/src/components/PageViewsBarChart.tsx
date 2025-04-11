import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSelector } from "react-redux";
import { RootState } from "../redux/store";
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import { BarChart } from '@mui/x-charts/BarChart';
import { useTheme } from '@mui/material/styles';

type StatCardProps = {
  id: string;
  label: string;
  data: number[];
  stack: string;
};

export default function PageViewsBarChart() {
  const theme = useTheme();
  const year = useSelector((state: RootState) => state.tapApp.year);
  const [total, setTotal] = useState(0);
  const [statData, setStatData] = useState<StatCardProps[]>(
  [
    {
      id: 'fatal',
      label: 'Fatal',
      data: [2234, 3872, 2998, 4125, 3357, 2789, 2998],
      stack: 'A',
    },
    {
      id: 'non-fatal',
      label: 'Non-Fatal',
      data: [3098, 4215, 2384, 2101, 4752, 3593, 2384],
      stack: 'A',
    }
  ]);
  const colorPalette = [
    (theme).palette.primary.dark,
    (theme).palette.primary.light,
  ];
  const apiUrl = 'http://127.0.0.1:5000'; 

  useEffect(() => {
      const fetchData = async () => {

        let year_total = 0;
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
          console.log('Fetched data:', data);

          const newData = [...statData];
          year_total += data.total;
          newData[0].data = data.per_month;
          setStatData(newData);

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
          console.log('Fetched data:', data);

          const newData = [...statData];
          year_total += data.total;
          newData[1].data = data.per_month;
          setStatData(newData);

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 

        setTotal(year_total);
      };
  
      fetchData();
    }, [year]);

  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
        <Typography component="h2" variant="subtitle2" gutterBottom>
          Month versus Accident Type
        </Typography>
        <Stack sx={{ justifyContent: 'space-between' }}>
          <Stack
            direction="row"
            sx={{
              alignContent: { xs: 'center', sm: 'flex-start' },
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Typography variant="h4" component="p">
              {total}
            </Typography>
            <Chip size="small" color="error" />
          </Stack>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Total Accidents in {year}
          </Typography>
        </Stack>
        <BarChart
          borderRadius={8}
          colors={colorPalette}
          xAxis={
            [
              {
                scaleType: 'band',
                categoryGapRatio: 0.5,
                data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'],
              },
            ] as any
          }
          series={statData}
          height={250}
          margin={{ left: 50, right: 0, top: 20, bottom: 20 }}
          grid={{ horizontal: true }}
          slotProps={{
            legend: {
              hidden: true,
            },
          }}
        />
      </CardContent>
    </Card>
  );
}
