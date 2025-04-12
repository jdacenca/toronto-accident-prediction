import * as React from "react";
import { useEffect, useState } from "react";
import Papa from "papaparse";
import { BarChart } from "@mui/x-charts/BarChart";
import { axisClasses } from "@mui/x-charts/ChartsAxis";
import { Box } from "@mui/material";

interface ModelPerformance {
  Classifier: string;
  ["Train Accuracy"]: number;
  ["Test Accuracy"]: number;
  ["Unseen Accuracy (10)"]: number;
  ["Unseen Accuracy (6000)"]: number;
  ["Precision"]: number;
  ["Recall"]: number;
  ["F1 Score"]: number;
}

const chartSetting = {
  yAxis: [
    {
      label: "Accuracy (%)",
    },
  ],
  width: 1000,
  height: 400,
  sx: {
    [`.${axisClasses.left} .${axisClasses.label}`]: {
      transform: "translate(-20px, 0)",
    },
  },
};

export default function AccuracyChart() {
  const [dataset, setDataset] = useState<any[]>([]);

  useEffect(() => {
    Papa.parse<ModelPerformance>(
      "/images/combined/classifier_performance.csv",
      {
        header: true,
        download: true,
        dynamicTyping: true,
        complete: (results) => {
          const cleaned = results.data
            .filter((item) => item.Classifier)
            .map((item) => ({
              Classifier: item.Classifier,
              "Train Accuracy": item["Train Accuracy"] * 100,
              "Test Accuracy": item["Test Accuracy"] * 100,
              "Unseen Accuracy (10)": item["Unseen Accuracy (10)"] * 100,
              "Unseen Accuracy (6000)": item["Unseen Accuracy (6000)"] * 100,
              Precision: item["Precision"] * 100,
              Recall: item["Recall"] * 100,
              "F1 Score": item["F1 Score"] * 100,
            }));
          setDataset(cleaned);
        },
      }
    );
  }, []);

  const valueFormatter = (value: number | null): string =>
    value !== null ? `${value.toFixed(1)}%` : "N/A";

  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between" }}>
        <BarChart
          dataset={dataset}
          xAxis={[
            { scaleType: "band", dataKey: "Classifier", label: "Classifier" },
          ]}
          series={[
            { dataKey: "Train Accuracy", label: "Train", valueFormatter },
            { dataKey: "Test Accuracy", label: "Test", valueFormatter },
            {
              dataKey: "Unseen Accuracy (10)",
              label: "Unseen (10)",
              valueFormatter,
            },
            {
              dataKey: "Unseen Accuracy (6000)",
              label: "Unseen (6000)",
              valueFormatter,
            },
            { dataKey: "Precision", label: "Precision", valueFormatter },
            { dataKey: "Recall", label: "Recall", valueFormatter },
            { dataKey: "F1 Score", label: "F1 Score", valueFormatter },
          ]}
          {...chartSetting}
        />
      </Box>
    </Box>
  );
}
