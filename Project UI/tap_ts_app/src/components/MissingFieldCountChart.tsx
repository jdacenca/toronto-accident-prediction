import * as React from 'react';
import { useState, useEffect } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import { BarChart } from '@mui/x-charts/BarChart';
import { useTheme } from '@mui/material/styles';

type StatCardProps = {
  data: (number | null)[];
};

export default function MissingFieldCountChart() {

  interface DatasetItem {
    name: string;
    value: number[];
  }
  
  const theme = useTheme();
  const [colData, setColData] = useState<DatasetItem[]>([]);
  const [statData, setStatData] = useState<StatCardProps[]>([
    {
    data: ['abcd'],
    }
  ]);


  const colorPalette = [
    (theme).palette.primary.dark, 
    (theme).palette.primary.main,
    (theme).palette.primary.light,
  ];
  const apiUrl = 'http://127.0.0.1:5000'; 

    useEffect(() => {
      const fetchData = async () => {

        // Fetch the total accidents for the selected year for Fatalities
        try {
          const response = await fetch(apiUrl + '/data/missing', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log('Fetched data:', data);

          if (!statData) {
            console.error('statData is undefined');
            return;
          }

          const dataset: DatasetItem[] = Object.entries(data).map(([name, value]) => ({
            name,
            value: Object.keys(data).map(key => Number(key)),
          }));

          setColData(dataset);

          console.log('MISSING Dataset:', dataset);
          
          const newData = [...statData];
          newData[0].data = Object.values(data).map(value => (typeof value === 'string' ? parseFloat(value) : value));
          setStatData(newData);
          console.log('MISSING NEW Dataset:', newData);

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 
      };
  
      fetchData();
    }, []);

  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
        <Typography component="h2" variant="subtitle2" gutterBottom>
          Missing Field Distribution
        </Typography>
        <Stack sx={{ justifyContent: 'space-between' }}>
          <BarChart
            dataset={colData}
            borderRadius={8}
            colors={colorPalette}
            yAxis={
              [
                {
                  scaleType: 'band',
                  categoryGapRatio: 0.5,
                  dataKey: 'name',
                },
              ] as any
            }
            xAxis={
              [
                {
                  scaleType: 'sqrt',
                },
              ] as any
            }
            layout="horizontal"
            series={statData}
            height={600}
            margin={{ left: 100, right: 0, top: 20, bottom: 20 }}
            grid={{ horizontal: true }}
            slotProps={{
              legend: {
                hidden: true,
              },
            }}
          />
        </Stack>
      </CardContent>
    </Card>
  );
}
