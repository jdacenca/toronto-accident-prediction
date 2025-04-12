import * as React from "react";
import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import { BarChart } from "@mui/x-charts/BarChart";
import { axisClasses } from "@mui/x-charts/ChartsAxis";
import Papa from "papaparse";
import { useTheme } from "@mui/material/styles";

type FeatureData = {
  feature: string;
  importance: number;
};

type FeatureImportanceChartProps = {
  model: string;
};

const chartSetting = {
  xAxis: [
    {
      label: "Importance",
      scaleType: "linear",
    },
  ],
  layout: "horizontal" as const,
  height: 600,
  margin: { top: 20, right: 30, bottom: 50, left: 145 },
  grid: { horizontal: true },
  slotProps: {
    legend: {
      hidden: true,
    },
  },
  sx: {
    [`.${axisClasses.bottom} .${axisClasses.label}`]: {
      transform: "translate(0, 10px)",
    },
    [`.${axisClasses.left} .${axisClasses.label}`]: {
      transform: "translate(-80px, 0)",
    },
  },
};

export default function FeatureImportanceVerticalChart({
  model,
}: FeatureImportanceChartProps) {
  const [dataset, setDataset] = useState<FeatureData[]>([]);
  const theme = useTheme();

  const colorPalette = [
    theme.palette.primary.dark,
    theme.palette.primary.main,
    theme.palette.primary.light,
  ];

  useEffect(() => {
    Papa.parse<FeatureData>(`/images/${model}/feature_importance.csv`, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete: (results) => {
        const data = results.data
          .filter((item) => item.feature && typeof item.importance === "number")
          .sort((a, b) => b.importance - a.importance);
        setDataset(data);
      },
    });
  }, [model]);

  const valueFormatter = (value: number | string | null) => {
    if (value === null) return "N/A";
    const num = typeof value === "number" ? value : parseFloat(value);
    return `${(num * 100).toFixed(1)}%`;
  };

  return (
    <Card variant="outlined" sx={{ width: "100%" }}>
      <CardContent>
        <Typography component="h2" variant="h6" gutterBottom>
          Feature Importance
        </Typography>
        <Stack sx={{ justifyContent: "space-between" }}>
          <BarChart
            dataset={dataset}
            borderRadius={8}
            colors={colorPalette}
            yAxis={[
              {
                scaleType: "band",
                categoryGapRatio: 0.5,
                dataKey: "feature",
              },
            ]}
            series={[
              {
                dataKey: "importance",
                valueFormatter,
              },
            ]}
            {...chartSetting}
          />
        </Stack>
      </CardContent>
    </Card>
  );
}
