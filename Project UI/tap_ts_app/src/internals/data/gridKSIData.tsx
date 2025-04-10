import * as React from 'react';
import Avatar from '@mui/material/Avatar';
import Chip from '@mui/material/Chip';
import { GridCellParams, GridRowsProp, GridColDef } from '@mui/x-data-grid';

function renderInjuryType(status: 'Fatal' | 'Non-Fatal Injury') {
  const colors: { [index: string]: 'success' | 'default' } = {
    "Fatal": 'success',
    "Non-Fatal Injury": 'default',
  };

  return <Chip label={status} color={colors[status]} size="small" />;
}

export function renderAvatar(
  params: GridCellParams<{ name: string; color: string }, any, any>,
) {
  if (params.value == null) {
    return '';
  }

  return (
    <Avatar
      sx={{
        backgroundColor: params.value.color,
        width: '24px',
        height: '24px',
        fontSize: '0.85rem',
      }}
    >
      {params.value.name.toUpperCase().substring(0, 1)}
    </Avatar>
  );
}

export const columns: GridColDef[] = [
  { field: 'Accnum', headerName: 'Id', flex: 0.5, minWidth: 10 },
  {
    field: 'Injury',
    headerName: 'Injury',
    flex: 1,
    minWidth: 50,
    renderCell: (params) => renderInjuryType(params.value as any)
  },
  {
    field: 'Neighborhood',
    headerName: 'Neighborhood',
    flex: 1.5,
    minWidth: 150
  },
  {
    field: 'Visibility',
    headerName: 'Visibility',
    flex: 1,
    minWidth: 80,
  },
  {
    field: 'Light',
    headerName: 'Light',
    flex: 1,
    minWidth: 100,
  },
  {
    field: 'RDSFCOND',
    headerName: 'Road Condition',
    flex: 1,
    minWidth: 120,
  }
];

export const rows: GridRowsProp = [
  {
    id: 1,
    pageTitle: 'Homepage Overview',
    status: 'Online',
    eventCount: 8345,
    users: 212423,
    viewsPerUser: 18.5,
    averageTime: '2m 15s',
    conversions: [
      469172, 488506, 592287, 617401, 640374, 632751, 668638, 807246, 749198, 944863,
      911787, 844815, 992022, 1143838, 1446926, 1267886, 1362511, 1348746, 1560533,
      1670690, 1695142, 1916613, 1823306, 1683646, 2025965, 2529989, 3263473,
      3296541, 3041524, 2599497,
    ],
  }
];
