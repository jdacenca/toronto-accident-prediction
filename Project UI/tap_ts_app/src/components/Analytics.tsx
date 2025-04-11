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
      trainingAccuracy: "91.52%",
      testingAccuracy: "89.58%",
      unseenDataAccuracy: "80%",
    },
    rf: {
      trainingAccuracy: "100%",
      testingAccuracy: "94.44%",
      unseenDataAccuracy: "90%",
    },
    svc: {
      trainingAccuracy: "99.46%",
      testingAccuracy: "94.68%",
      unseenDataAccuracy: "90%",
    },
    dt: {
      trainingAccuracy: "100%",
      testingAccuracy: "90.74%",
      unseenDataAccuracy: "0.7%",
    },
    nn: {
      trainingAccuracy: "97.93%",
      testingAccuracy: "91.15%",
      unseenDataAccuracy: "90%",
    },
    hv: {
      trainingAccuracy: "99.81%",
      testingAccuracy: "94.39%",
      unseenDataAccuracy: "80%",
    },
    sv: {
      trainingAccuracy: "99.68%",
      testingAccuracy: "94.21%",
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
    <Box sx={{ flexGrow: 1, padding: 3 }}>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, lg: 12 }}>
          <Typography variant="h4" component="h2">
            Model Analytics
          </Typography>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Stack direction="row" sx={{ gap: 1 }}>
            <Typography component="h4" variant="h4" sx={{ mb: 2 }}>
              Select Model:
            </Typography>
            <FormControl sx={{ m: 0, minWidth: 150 }}>
              <Select
                labelId="select-model"
                id="select-model"
                value={model}
                label="Model"
                onChange={handleChange}
                MenuProps={{
                  PaperProps: {
                    style: {
                      maxHeight: 250, // Adjust this value as needed
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
        </Grid>

        <Grid size={{ xs: 12, md: 8 }}>
          <Box
            sx={{
              display: "grow-flex",
              gap: 5,
              width: "100%",
              alignItems: "center",
            }}
          >
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  {getModelData(model).trainingAccuracy}
                </Typography>
                <Typography variant="subtitle2">Training Accuracy</Typography>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  {getModelData(model).testingAccuracy}
                </Typography>
                <Typography variant="subtitle2">Testing Accuracy</Typography>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">
                  {getModelData(model).unseenDataAccuracy}
                </Typography>
                <Typography variant="subtitle2">
                  Unseen Data Accuracy
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
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" component="h3">
                Feature Importance
              </Typography>
              <FeatureImportanceChart model={model} />
            </CardContent>
          </Card>
        </Grid>

        {/* <Grid size={{ xs: 12 }}>
          <Typography variant="h6" component="h3">
            Performance Analysis
          </Typography>
          <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label="performance table">
              <TableHead>
                <TableRow>
                  <TableCell>Kernel Type</TableCell>
                  <TableCell>C Value</TableCell>
                  <TableCell>Gamma</TableCell>
                  <TableCell>Degree</TableCell>
                  <TableCell>CV Scores (5 folds)</TableCell>
                  <TableCell>Mean Score</TableCell>
                  <TableCell>Best Parameters</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Linear</TableCell>
                  <TableCell>1</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>[0.964, 0.982, 0.964, 0.955, 0.973]</TableCell>
                  <TableCell>0.9678</TableCell>
                  <TableCell>svm_C=1, svm_kernel=linear</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>RBF</TableCell>
                  <TableCell>0.1</TableCell>
                  <TableCell>0.03</TableCell>
                  <TableCell>-</TableCell>
                  <TableCell>[0.973, 0.982, 0.955, 0.955, 0.973]</TableCell>
                  <TableCell>0.9676</TableCell>
                  <TableCell>svm_C=0.1, svm_gamma=0.03</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Polynomial</TableCell>
                  <TableCell>1</TableCell>
                  <TableCell>1.0</TableCell>
                  <TableCell>3</TableCell>
                  <TableCell>[0.946, 0.973, 0.955, 0.920, 0.946]</TableCell>
                  <TableCell>0.9480</TableCell>
                  <TableCell>svm_C=1, svm_degree=3, svm_gamma=1.0</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Grid> */}
      </Grid>
    </Box>
  );
}

export default Analytics;
