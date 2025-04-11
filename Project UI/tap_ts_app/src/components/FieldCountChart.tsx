import * as React from 'react';
import { useState, useEffect } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import { BarChart } from '@mui/x-charts/BarChart';
import { useTheme } from '@mui/material/styles';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';

type StatCardProps = {
  data: number[];
};

export default function FieldCountChart() {

  interface ApiResponseItem {
    value: string;
    label: string;
  }

  interface DatasetItem {
    name: string;
    value: number;
  }
  
  const theme = useTheme();
  const [columns, setColumns] = useState<ApiResponseItem[]>([]);
  const [field, setField] = React.useState('DISTRICT');
  const [colData, setColData] = useState<DatasetItem[]>([]);
  const [statData, setStatData] = useState<StatCardProps[]>([]);

  const handleChange = (event: SelectChangeEvent) => {
    setField(event.target.value);
  };

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
          const response = await fetch(apiUrl + '/data/distribution_list', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data: string[] = await response.json();
          setColumns(data.map(item => ({ value: item, label: item })));

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 
      };
  
      fetchData();
    }, []);

    useEffect(() => {
      const fetchData = async () => {

        // Fetch the total accidents for the selected year for Fatalities
        try {
          const response = await fetch(apiUrl + '/data/distribution/' + field, {
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

          const dataset: DatasetItem[] = Object.entries(data.labels).map(([name, value]) => ({
            name: String(value),
            value: [],
          }));

          setColData(dataset);

          console.log('Dataset:', dataset);

          const stat: StatCardProps[] = Object.entries(data.data).map(([name, value]) => ({
            data: Array.isArray(value) && value.every(item => typeof item === 'number') ? value : [],
          }));
          
          console.log(stat);
          setStatData(stat);

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 
      };
  
      fetchData();
    }, [field]);

  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
        <Typography component="h2" variant="subtitle2" gutterBottom>
          Field Distribution
        </Typography>
        <Stack sx={{ justifyContent: 'space-between' }}>
          <FormControl sx={{ m: 0, maxWidth: 200}}>
            <Select
              labelId="select-field"
              id="select-field"
              value={field}
              label="Year"
              onChange={handleChange}
              MenuProps={{
                  PaperProps: {
                    style: {
                      maxHeight: 200, // Adjust this value as needed
                    },
                  },
                }}
            >
              <MenuItem value="DISTRICT">
              </MenuItem>
              {columns.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                      {String(option.label)}
                  </MenuItem>
              ))}
            </Select>
          </FormControl>
          <BarChart
            dataset={colData as DatasetItem[]}
            borderRadius={8}
            colors={colorPalette}
            xAxis={
              [
                {
                  scaleType: 'band',
                  categoryGapRatio: 0.5,
                  dataKey: 'name',
                },
              ] as any
            }
            yAxis={[
              {
                scaleType: 'sqrt',
                
              },
            ]}
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
        </Stack>
      </CardContent>
    </Card>
  );
}
