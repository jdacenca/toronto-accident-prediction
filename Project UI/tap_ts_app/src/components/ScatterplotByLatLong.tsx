import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSelector } from "react-redux";
import { RootState } from "../redux/store";
import { useTheme } from '@mui/material/styles';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import { ScatterChart } from '@mui/x-charts/ScatterChart';

type LatLong = {
    id: number;
    x: number;
    y: number;
}

function transformData(data: any): LatLong[] {
  const latData = data['LAT'];
  const longData = data.LONG;
  const id = data.id;
  const result: LatLong[] = [];

  // Assuming the keys in both LAT and LONG objects align
  for (const key in latData) {

    if (longData.hasOwnProperty(key)) {
      result.push({ id: id[key], x: longData[key], y: latData[key] }); // Note: x is LONG, y is LAT
    }
  }

  return result;
}
export default function ScatterplotByLatLong() {
    const theme = useTheme();
    const year = useSelector((state: RootState) => state.tapApp.year);
    const [fatalStatData, setFatalStatData] = useState<LatLong[]>([]);
    const [nonFatalStatData, setNonFatalStatData] = useState<LatLong[]>([]);
    const apiUrl = 'http://127.0.0.1:5000'; 
    
    useEffect(() => {
      const fetchData = async () => {

        // Fetch the total accidents for the selected year for Fatalities
        try {
          const response = await fetch(apiUrl + '/data/latlong/' + year + '/Fatal', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data: any = await response.json();

          let newFatalData = fatalStatData;
          const processedData: LatLong[] = transformData(data);
          newFatalData = processedData;
          setFatalStatData(newFatalData);

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 

        // Fetch the total accidents for the selected year for Non Fatalities
        try {
          const response = await fetch(apiUrl + '/data/latlong/' + year + '/Non-Fatal%20Injury', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data: any = await response.json();

          let newNonFatalData = nonFatalStatData;
          const processedData: LatLong[] = transformData(data);
          newNonFatalData = processedData;
          setNonFatalStatData(newNonFatalData);

        } catch (err) {
          console.error('Error fetching data:', err);
          console.error('Failed to load options. Please try again later.');
        } 
      };
      fetchData();
    }, [year]);
  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
          <CardContent>
            <Typography component="h2" variant="subtitle2" gutterBottom>
              Scatter Plot of Latitude and Longitude
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
              </Stack>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Latitude and Longitude in {year} 
              </Typography>
            </Stack>
            <ScatterChart
                width={600}
                height={285}
                series={[
                  {
                    label: 'Fatal',
                    data: fatalStatData.map((v) => ({ x: v.x, y: v.y, id: v.id })),
                    color: 'rgba(224, 152, 145, 0.6)',
                  },
                  {
                    label: 'Non-Fatal',
                    data: nonFatalStatData.map((v) => ({ x: v.x, y: v.y, id: v.id })),
                    color: 'rgba(133, 186, 161, 0.6)',
                  },
                ]}
              />
          </CardContent>
        </Card>




    
  );
}