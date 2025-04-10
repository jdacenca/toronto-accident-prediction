import * as React from 'react';
import { useState, useEffect } from 'react';
import { useSelector } from "react-redux";
import { RootState } from "../redux/store";
import { DataGrid } from '@mui/x-data-grid';
import { GridRowsProp } from '@mui/x-data-grid';
import { columns } from '../internals/data/gridKSIData';

export default function CustomizedDataGrid() {
  const [rows, setRows] = useState<GridRowsProp>([]);
  const year = useSelector((state: RootState) => state.tapApp.year);
  const apiUrl = 'http://127.0.0.1:5000';

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(apiUrl + '/data/base/' + year, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: any = await response.json();
        console.log('ROWS:', data);
        setRows(data);

      } catch (err) {
        console.error('Error fetching data:', err);
        console.error('Failed to load options. Please try again later.');
      } 
    };

    fetchData();
  }, [year]);

  return (
    <DataGrid
      rows={rows}
      columns={columns}
      getRowClassName={(params) =>
        params.indexRelativeToCurrentPage % 2 === 0 ? 'even' : 'odd'
      }
      initialState={{
        pagination: { paginationModel: { pageSize: 20 } },
      }}
      pageSizeOptions={[10, 20, 50]}
      disableColumnResize
      density="compact"
      slotProps={{
        filterPanel: {
          filterFormProps: {
            logicOperatorInputProps: {
              variant: 'outlined',
              size: 'small',
            },
            columnInputProps: {
              variant: 'outlined',
              size: 'small',
              sx: { mt: 'auto' },
            },
            operatorInputProps: {
              variant: 'outlined',
              size: 'small',
              sx: { mt: 'auto' },
            },
            valueInputProps: {
              InputComponentProps: {
                variant: 'outlined',
                size: 'small',
              },
            },
          },
        },
      }}
    />
  );
}
