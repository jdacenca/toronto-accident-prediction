import React, { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import {
  DataGrid,
  GridColDef,
  GridRenderCellParams,
  GridToolbar,
} from "@mui/x-data-grid";
import Link from "@mui/material/Link";
import Papa from "papaparse";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";

function ExpandableCell({ value }: GridRenderCellParams) {
  const [expanded, setExpanded] = React.useState(false);

  return (
    <div>
      {expanded ? value : String(value).slice(0, 200)}&nbsp;
      {String(value).length > 200 && (
        <Link
          type="button"
          component="button"
          sx={{ fontSize: "inherit", letterSpacing: "inherit" }}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "view less" : "view more"}
        </Link>
      )}
    </div>
  );
}

const columns: GridColDef[] = [
  { field: "Classifier", headerName: "Classifier", width: 200 },
  {
    field: "Train Accuracy",
    headerName: "Train Acc. %",
    width: 120,
    type: "number",
  },
  {
    field: "Test Accuracy",
    headerName: "Test Acc. %",
    width: 120,
    type: "number",
  },
  {
    field: "Unseen Accuracy (10)",
    headerName: "Unseen Acc. %",
    width: 120,
    type: "number",
  },
  { field: "Precision", headerName: "Precision", width: 120, type: "number" },
  { field: "Recall", headerName: "Recall", width: 120, type: "number" },
  { field: "F1 Score", headerName: "F1-Score", width: 100, type: "number" },
  {
    field: "Unseen Accuracy (6000)",
    headerName: "Unseen Acc. %",
    width: 120,
    type: "number",
  },
  { field: "ROC AUC", headerName: "ROC AUC", width: 120, type: "number" },
];

interface ModelPerformanceGridProps {
  model: string;
}

export default function ModelPerformanceGrid({
  model,
}: ModelPerformanceGridProps) {
  const [rows, setRows] = useState<any[]>([]);

  useEffect(() => {
    Papa.parse(`/images/${model}/model_performance.csv`, {
      header: true,
      download: true,
      dynamicTyping: true,
      skipEmptyLines: true, // skip blank rows
      complete: (result) => {
        if (result.errors.length > 0) {
          console.error("Parsing errors:", result.errors);
        } else {
          const parsedRows = result.data.map((row: any, index: number) => ({
            id: row.id || index + 1,
            ...row,
          }));
          setRows(parsedRows);
        }
      },
    });
  }, [model]);

  return (
    <DataGrid
      rows={rows}
      columns={columns}
      getRowHeight={() => "auto"}
      slots={{ toolbar: GridToolbar }}
      initialState={{
        pagination: {
          paginationModel: { pageSize: 5 },
        },
      }}
      pageSizeOptions={[5]}
      sx={{
        height: "400px",
        width: "1080px", // Set a fixed width for the DataGrid
        border: "1px solid #ccc",
        "&.MuiDataGrid-root--densityCompact .MuiDataGrid-cell": { py: 1 },
        "&.MuiDataGrid-root--densityStandard .MuiDataGrid-cell": { py: "15px" },
        "&.MuiDataGrid-root--densityComfortable .MuiDataGrid-cell": {
          py: "22px",
        },
      }}
    />
  );
}
