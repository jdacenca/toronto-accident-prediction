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
//  Kernel,Train Acc.,Test Acc.,Unseen Acc.,Parameters,Precision,F1-Score,Recall,ROC Score,Class Imbalance
{field: "id", headerName: "ID", width: 70},
{field: "Train Acc.", headerName: "Train Acc.%", width: 110,type: "number"},
{field: "Test Acc.", headerName: "Test Acc.%", width: 110,type: "number"},
{field: "Unseen Acc.", headerName: "Unseen Acc.%", width: 120,type: "number"},
{field: "Parameters", headerName: "Parameters", width: 150},
{field: "Precision", headerName: "Precision", width: 110,type: "number"},
{field: "F1-Score", headerName: "F1-Score", width: 100,type: "number"},
{field: "Recall", headerName: "Recall", width: 100,type: "number"},
{field: "ROC Score", headerName: "ROC Score", width: 100,type: "number"},
{field: "Class Imbalance", headerName: "Class Imbalance", width: 160},

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
          paginationModel: { pageSize: 12 },
        },
      }}
      pageSizeOptions={[12]}
      sx={{
        height: "700px",
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
