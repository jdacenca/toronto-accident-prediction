import * as React from 'react';
import Stack from '@mui/material/Stack';
import NavbarBreadcrumbs from './NavbarBreadcrumbs';

export default function Header() {
  return (
    <>
    {/*add margin5 px*/}
    <Stack
      direction="row"
      spacing={2}
      sx={{
        width: '100%',
        maxWidth: { sm: '100%', md: '1700px', lg: '1400px' },
        mx: 5,
        mb: 2,
      }}
    >
      </Stack>
 

    </>
  );
}
