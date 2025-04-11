import * as React from 'react';
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import { BarChart } from '@mui/x-charts/BarChart';
import { axisClasses } from '@mui/x-charts/ChartsAxis';

import Papa from 'papaparse';

type FeatureData = {
  feature: string;
  importance: number;
};

type FeatureImportanceChartProps = {
  model: string;
};

const chartSetting = {
  yAxis: [
    {
      label: 'Importance',
    },
  ],
  width: 1000,
  height: 400,
  sx: {
    [`.${axisClasses.left} .${axisClasses.label}`]: {
      transform: 'translate(-15px, 0)',
    },
  },
};

export default function FeatureImportanceChart({ model }: FeatureImportanceChartProps) {
  const [dataset, setDataset] = useState<FeatureData[]>([]);
  const [itemNb, setItemNb] = useState(30);

  useEffect(() => {
    Papa.parse<FeatureData>(`/images/${model}/feature_importance.csv`, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (results) => {
        const data = results.data
          .filter((item) => item.feature && typeof item.importance === 'number')
          .sort((a, b) => b.importance - a.importance);
        setDataset(data);
      },
    });
  }, [model]);

  const valueFormatter = (value: number | string | null) => {
    if (value === null) return 'N/A';
    const num = typeof value === 'number' ? value : parseFloat(value);
    return `${(num * 100).toFixed(1)}%`;
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
      <BarChart
        dataset={dataset.slice(0, itemNb)}
        xAxis={[{ scaleType: 'band', dataKey: 'feature', label: 'Feature' }]}
        series={[
          {
            dataKey: 'importance',
            label: 'Importance',
            valueFormatter,
          },
        ]}

        {...chartSetting}
      />
      </Box>
      <Typography id="input-item-number" gutterBottom sx={{ mt: 2 }}>
        Number of features to display
      </Typography>
      <Slider
        value={itemNb}
        onChange={(_, newValue) => setItemNb(newValue as number)}
        valueLabelDisplay="auto"
        min={1}
        max={dataset.length || 0}
     
        aria-labelledby="input-item-number"
      />
    </Box>
  );
}
