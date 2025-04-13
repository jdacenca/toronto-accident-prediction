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
import { AppDispatch, RootState } from "../redux/store";
import { setModel } from "../redux/slice";
import FeatureImportanceChart from "./FeatureImportanceChart.tsx";
import FeatureImportanceVerticalChart from "./FeatureImportanceVerticalChart.tsx";
import ModelPerformanceGrid from "./ModelPerformanceGrid.tsx";

function Analytics() {
  const model = useSelector((state: RootState) => state.tapApp.model);

  const dispatch = useDispatch<AppDispatch>();

  const modelResultsMap: Record<
    string,
    {
      trainingAccuracy: string;
      testingAccuracy: string;
      unseenDataAccuracy: string;
    }
  > = {
    lg: {
      trainingAccuracy: "91.62%",
      testingAccuracy: "89.58%",
      unseenDataAccuracy: "80%",
    },
    rf: {
      trainingAccuracy: "100%",
      testingAccuracy: "94.10%",
      unseenDataAccuracy: "90%",
    },
    svc: {
      trainingAccuracy: "99.54%",
      testingAccuracy: "94.79%",
      unseenDataAccuracy: "100%",
    },
    dt: {
      trainingAccuracy: "97.21%",
      testingAccuracy: "95.27%",
      unseenDataAccuracy: "90%",
    },
    nn: {
      trainingAccuracy: "97.68%",
      testingAccuracy: "92.36%",
      unseenDataAccuracy: "90%",
    },
    hv: {
      trainingAccuracy: "99.71%",
      testingAccuracy: "94.16%",
      unseenDataAccuracy: "90%",
    },
    sv: {
      trainingAccuracy: "99.99%",
      testingAccuracy: "95%",
      unseenDataAccuracy: "80%",
    },
  };

  const getModelData = (model: string) => {
    return modelResultsMap[model];
  };

  const handleChange = (event: SelectChangeEvent) => {
    setModel(event.target.value);
    dispatch(setModel(String(event.target.value)));
  };

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
            Model Analytics
          </Typography>
        </Grid>

        <Grid size={{ xs: 12, md: 3.5 }}>
          <Box
            sx={{
              display: "grow-flex",
              gap: 5,
              width: "100%",
              alignItems: "center",
            }}
          >
            <Stack direction="row" sx={{ gap: 1 }}>
              <Typography component="h6" variant="h6" sx={{ mb: 2 }}>
                Select Model:
              </Typography>
              <FormControl variant="outlined" sx={{ m: 0, minWidth: 165 }}>
                <Select
                  labelId="select-model"
                  id="select-model"
                  value={model}
                  label="Model"
                  variant="outlined"
                  onChange={handleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem key="lg" value="lg">
                    Logistic Regression
                  </MenuItem>
                  <MenuItem key="rf" value="rf">
                    Random Forest
                  </MenuItem>
                  <MenuItem key="svc" value="svc">
                    SVC
                  </MenuItem>
                  <MenuItem key="dt" value="dt">
                    Decision Tree
                  </MenuItem>
                  <MenuItem key="nn" value="nn">
                    MLP Classifier
                  </MenuItem>
                  <MenuItem key="hv" value="hv">
                    Hard Voting
                  </MenuItem>
                  <MenuItem key="sv" value="sv">
                    Soft Voting
                  </MenuItem>
                </Select>
              </FormControl>
            </Stack>
          </Box>
        </Grid>

        <Grid size={{ xs: 12, md: 8.5 }}>
          <Box
            sx={{
              display: "grow-flex",
              gap: 2,
              width: "100%",
              alignItems: "center",
            }}
          >
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  Training Accuracy: {getModelData(model).trainingAccuracy}
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  Testing Accuracy: {getModelData(model).testingAccuracy}
                </Typography>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  Unseen Acc: {getModelData(model).unseenDataAccuracy}
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" component="h3">
                Confusion Matrix
              </Typography>
              <img
                src={`/images/${model}/confusion_matrix.png`}
                alt="Confusion Matrix"
                style={{ width: "100%", height: "auto" }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" component="h3">
                Roc Curve
              </Typography>
              <img
                src={`/images/${model}/roc_curve.png`}
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
          <FeatureImportanceVerticalChart model={model} />
        </Grid>

        <Typography variant="h6">
          Performance Comparison Across Various Sampling Techniques
        </Typography>
        <Grid size={{ xs: 12, lg: 12 }}>
          <ModelPerformanceGrid model={model} />
        </Grid>
      </Grid>
    </Box>
  );
}

export default Analytics;
