import {
  Box,
  Card,
  Stack,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
} from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch, RootState } from "../redux/store.tsx";
import { setModel } from "../redux/slice.tsx";
import AccuracyChart from "./AccuracyChart.tsx";
import { useEffect, useState } from "react";
import PerformanceComparisonGrid from "./PerformanceComparisonGrid.tsx";

function ModelComparison() {
  const model = useSelector((state: RootState) => state.tapApp.model);

  const dispatch = useDispatch<AppDispatch>();

  return (
    <Box
      sx={{
        width: "100%",
        maxWidth: { sm: "100%", md: "1700px", lg: "1400px" },
      }}
    >
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 12 }}>
          <Typography variant="h4" component="h2">
            Model Comparison
          </Typography>
        </Grid>
        <Grid size={{ xs: 12 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" component="h3">
                Performance
              </Typography>
              <AccuracyChart />
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <img
                src={`/images/combined/combined_roc_curve_testing.png`}
                alt="Conmbined_roc"
                style={{ width: "100%", height: "auto" }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <img
                src={`/images/combined/combined_pr_curve_testing.png`}
                alt="Roc Curve"
                style={{ width: "100%", height: "auto" }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" component="h3">
                Classification Report
              </Typography>
              <img
                src={`/images/${model}/classification_report.png`}
                alt="Classification Report"
                style={{ width: "100%", height: "auto" }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12 }}>
          <PerformanceComparisonGrid />
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelComparison;
