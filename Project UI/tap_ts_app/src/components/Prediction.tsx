import { useState, useEffect, useCallback } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CardMedia from '@mui/material/CardMedia';
import Avatar from '@mui/material/Avatar';

import DT from '../assets/decision-tree.png';
import RF from '../assets/forest.png';
import LR from '../assets/logistic-regression.png';
import SVM from '../assets/scatter-graph.png';
import NN from '../assets/deep-learning.png';
import P from '../assets/predictive.png';

import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';
import OutlinedInput from '@mui/material/OutlinedInput';
import { styled } from '@mui/material/styles';

import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';

import AutoCompleteResult from './AutoCompleteResult';
import AutoCompleteControl from './AutoCompleteControl';

const FormGrid = styled(Grid)(() => ({
  display: 'flex',
  flexDirection: 'column',
}));

import {
  useMapsLibrary,
  AdvancedMarker,
  APIProvider,
  ControlPosition,
  Map,
  MapMouseEvent
} from '@vis.gl/react-google-maps';

const API_KEY = "";
export type AutocompleteMode = { id: string; label: string };

export default function Prediction() {
  const apiUrl = 'http://127.0.0.1:5000';

  const maps = useMapsLibrary('maps'); // Access the core maps library
  const places = useMapsLibrary('places'); // Access the places library

  const [placesService, setPlacesService] = useState<google.maps.places.PlacesService | null>(null);


  const roadClassRef = ['MAJOR ARTERIAL', 'MINOR ARTERIAL', 'COLLECTOR', 'LOCAL', 'OTHER', 'PENDING', 'LANEWAY', 'EXPRESSWAY', 'EXPRESSWAY RAMP', 'MAJOR SHORELINE'];
  const districtRef = ['TORONTO AND EAST YORK', 'NORTH YORK', 'SCARBOROUGH', 'ETOBICOKE YORK', 'OTHER'];
  const collitionLocationRef = ['AT INTERSECTION', 'AT/NEAR PRIVATE DRIVE', 'INTERSECTION RELATED', 'LANEWAY', 'NON INTERSECTION', 'OTHER', 'OVERPASS OR BRIDGE', 'UNDERPASS OR TUNNEL'];
  const trafficRef = ['NO CONTROL', 'TRAFFIC SIGNAL', 'PEDESTRIAN CROSSOVER', 'STOP SIGN', 'OTHER', 'YIELD SIGN', 'TRAFFIC CONTROLLER', 'SCHOOL GUARD', 'POLICE CONTROL', 'TRAFFIC GATE', 'STREETCAR (STOP FOR)'];
  const visibilityRef = ['CLEAR', 'SNOW', 'OTHER', 'RAIN', 'STRONG WIND', 'FOG, MIST, SMOKE, DUST', 'DRIFTING SNOW', 'FREEZING RAIN'];
  const lightRef = ['DARK', 'DARK, ARTIFICIAL', 'DAYLIGHT', 'DUSK', 'DAWN', 'DUSK, ARTIFICIAL', 'DAWN, ARTIFICIAL', 'DAYLIGHT, ARTIFICIAL', 'OTHER'];
  const roadSurfaceRef = ['WET', 'SLUSH', 'DRY', 'ICE', 'LOOSE SNOW', 'OTHER', 'PACKED SNOW', 'SPILLED LIQUID', 'LOOSE SAND OR GRAVEL'];
  const impactTypeRef = ['APPROACHING', 'SMV OTHER', 'PEDESTRIAN COLLISIONS', 'ANGLE', 'TURNING MOVEMENT', 'CYCLIST COLLISIONS', 'REAR END', 'SIDESWIPE', 'SMV UNATTENDED VEHICLE', 'OTHER'];
  const invtypeRef = ['PASSENGER', 'DRIVER', 'VEHICLE OWNER', 'OTHER PROPERTY OWNER', 'PEDESTRIAN', 'CYCLIST', 'OTHER', 'MOTORCYCLE DRIVER', 'TRUCK DRIVER', 'IN-LINE SKATER', 'DRIVER - NOT HIT', 'MOTORCYCLE PASSENGER', 'MOPED DRIVER', 'WHEELCHAIR', 'PEDESTRIAN - NOT HIT', 'TRAILER OWNER', 'WITNESS', 'CYCLIST PASSENGER', 'MOPED PASSENGER'];
  const ageRef = ['UNKNOWN', '0 TO 4', '5 TO 9', '10 TO 14', '15 TO 19', '20 TO 24', '25 TO 29', '30 TO 34', '35 TO 39', '40 TO 44', '45 TO 49', '50 TO 54', '55 TO 59', '60 TO 64', '65 TO 69', '70 TO 74', '75 TO 79', '80 TO 84', '85 TO 89', '90 TO 94', 'OVER 95'];
  const pedestrianRef = ['NA', 'INATTENTIVE', 'NORMAL', 'UNKNOWN', 'MEDICAL OR PHYSICAL DISABILITY', 'HAD BEEN DRINKING', 'ABILITY IMPAIRED, ALCOHOL', 'OTHER', 'ABILITY IMPAIRED, ALCOHOL OVER .80', 'ABILITY IMPAIRED, DRUGS', 'FATIGUE'];
  const cyclistRef = ['NA', 'NORMAL', 'INATTENTIVE', 'HAD BEEN DRINKING', 'UNKNOWN', 'ABILITY IMPAIRED, DRUGS', 'ABILITY IMPAIRED, ALCOHOL OVER .80', 'MEDICAL OR PHYSICAL DISABILITY', 'ABILITY IMPAIRED, ALCOHOL', 'OTHER', 'FATIGUE'];

  const setRoadClasshandleChange = (event: SelectChangeEvent) => {
    setRoadClass(event.target.value);
  };
  const setDistricthandleChange = (event: SelectChangeEvent) => {
    setDistrict(event.target.value);
  };
  const setcollitionLocationhandleChange = (event: SelectChangeEvent) => {
    setcollitionLocation(event.target.value);
  };
  const setTraffichandleChange = (event: SelectChangeEvent) => {
    setTraffic(event.target.value);
  };
  const setVisibilityhandleChange = (event: SelectChangeEvent) => {
    setVisibility(event.target.value);
  };
  const setLighthandleChange = (event: SelectChangeEvent) => {
    setLight(event.target.value);
  };
  const setRoadSurfaceLocationhandleChange = (event: SelectChangeEvent) => {
    setRoadSurface(event.target.value);
  };
  const setImpactTypehandleChange = (event: SelectChangeEvent) => {
    setImpactType(event.target.value);
  };
  const setInvtypehandleChange = (event: SelectChangeEvent) => {
    setInvtype(event.target.value);
  };
  const setAgehandleChange = (event: SelectChangeEvent) => {
    setAge(event.target.value);
  };
  const setPedestrianhandleChange = (event: SelectChangeEvent) => {
    setPedestrian(event.target.value);
  };
  const setCyclisthandleChange = (event: SelectChangeEvent) => {
    setCyclist(event.target.value);
  };


  const setisPedestrianhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisPedestrian(event.target.checked);
  };
  const setisCyclisthandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisCyclist(event.target.checked);
  };
  const setisAutomobilehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAutomobile(event.target.checked);
  };
  const setisMotorcyclehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisMotorcycle(event.target.checked);
  };
  const setisSpeedinghandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisSpeeding(event.target.checked);
  };
  const setisAggressivehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAggressive(event.target.checked);
  };
  const setisRedLighthandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisRedLight(event.target.checked);
  };
  const setisTruckhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisTruck(event.target.checked);
  };
  const setisCityVehiclehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisCityVehicle(event.target.checked);
  };
  const setisEmergencyhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisEmergency(event.target.checked);
  };
  const setisPassengerhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisPassenger(event.target.checked);
  };
  const setisAlcoholhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAlcohol(event.target.checked);
  };
  const setisDisabilityhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisDisability(event.target.checked);
  };

  const [lat, setLat] = useState(0);
  const [long, setLong] = useState(0);

  const mapClick = useCallback((ev: MapMouseEvent) => {
    //if(!maps) return;
    //if(!ev.latLng) return;
    if (ev.detail.latLng) {
      console.log('marker clicked:', ev.detail.latLng);
      setLat(ev.detail.latLng.lat);
      setLong(ev.detail.latLng.lng);

    } else {
      console.log('marker clicked: latLng is null');
    }
    //maps.panTo(ev.latLng);
  }, []);

  const handlePrediction = () => {
    console.log("PREDICTION")
  };
  useEffect(() => {
    if (maps && places) {
      const mapElement = document.createElement('div'); // Create a dummy div for PlacesService
      setPlacesService(new places.PlacesService(mapElement));
    }
  }, [maps, places]);

  const [isPedestrian, setisPedestrian] = useState(false);
  const [isCyclist, setisCyclist] = useState(false);
  const [isAutomobile, setisAutomobile] = useState(false);
  const [isMotorcycle, setisMotorcycle] = useState(false);
  const [isSpeeding, setisSpeeding] = useState(false);
  const [isAggressive, setisAggressive] = useState(false);
  const [isRedLight, setisRedLight] = useState(false);
  const [isTruck, setisTruck] = useState(false);
  const [isCityVehicle, setisCityVehicle] = useState(false);
  const [isEmergency, setisEmergency] = useState(false);
  const [isPassenger, setisPassenger] = useState(false);
  const [isAlcohol, setisAlcohol] = useState(false);
  const [isDisability, setisDisability] = useState(false);

  const [roadClass, setRoadClass] = useState('LOCAL');
  const [district, setDistrict] = useState('NORTH YORK');
  const [collitionLocation, setcollitionLocation] = useState('AT INTERSECTION');
  const [traffic, setTraffic] = useState('NO CONTROL');
  const [visibility, setVisibility] = useState('CLEAR');
  const [light, setLight] = useState('DARK');
  const [roadSurface, setRoadSurface] = useState('DRY');
  const [impactType, setImpactType] = useState('APPROACHING');
  const [invtype, setInvtype] = useState('DRIVER');
  const [age, setAge] = useState('UNKNOWN');
  const [pedestrian, setPedestrian] = useState('NORMAL');
  const [cyclist, setCyclist] = useState('NORMAL');

  const [selectedPlace, setSelectedPlace] =
    useState<google.maps.places.Place | null>(null);

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px', lg: '2400px' } }}>
      <Stack
        sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 3 }}
      >
        <Button
          onClick={handlePrediction}
          variant="outlined"
          sx={{ display: { xs: '12', m: '6' }, width: 500, alignItems: 'center', height: 50 }}
          startIcon={<Avatar src={P} />}
        >
          <Typography variant="body2" sx={{
            color: 'text.secondary', fontSize: 20,
            fontWeight: 700,
          }}>PREDICT</Typography>
        </Button>
      </Stack>
      <Grid
        container
        sx={{
          height: {
            xs: '100%'
          },
          mt: {
            xs: 4,
            sm: 0,
          },
        }}
      >

        <Grid
          container
          size={{ xs: 12, sm: 5, lg: 5 }}
          sx={{
            display: { xs: 'none', md: 'flex' },
            flexDirection: 'column',
            backgroundColor: 'background.paper',
            border: { sm: 'none', md: '1px solid' },
            borderColor: { sm: 'none', md: 'divider' },
            alignItems: 'start',
            pt: 5,
            px: 5,
            gap: 4,
          }}
        >
          <Card variant="outlined" sx={{ width: '100%' }}>
            <CardContent>
              <Stack sx={{ justifyContent: 'space-between' }}>
                <Stack
                  direction="row"
                  sx={{ justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <Typography variant="h4" component="p">
                    Location
                  </Typography>
                </Stack>
                <FormGrid size={{ xs: 12 }}>
                  <FormLabel htmlFor="latitude" required>
                    Latitude
                  </FormLabel>
                  <OutlinedInput
                    id="latitude"
                    name="latitude"
                    type="latitude"
                    required
                    size="small"
                    value={lat}
                  />
                </FormGrid>
                <FormGrid size={{ xs: 12 }}>
                  <FormLabel htmlFor="longitude" required>
                    Longitude
                  </FormLabel>
                  <OutlinedInput
                    id="longitude"
                    name="longitude"
                    type="longitude"
                    required
                    size="small"
                    value={long}
                  />
                </FormGrid>
              </Stack>
            </CardContent>
          </Card>
          <Stack
            direction="row"
            sx={{ justifyContent: 'space-between', alignItems: 'center', 'padding-top': 10 }}
          >
            <APIProvider apiKey={API_KEY} libraries={['marker']} solutionChannel='GMP_devsite_samples_v3_rgmbasicmap' onLoad={() => console.log('Maps API has loaded.')}>
              <Map
                mapId='Basic_Map'
                style={{ width: '30vw', height: '30vh' }}
                defaultZoom={11}
                defaultCenter={{ lat: 43.6532, lng: -79.3832 }}
                gestureHandling={'greedy'}
                disableDefaultUI={true}
                onClick={mapClick}
              >
              </Map>
              <AutoCompleteControl
                controlPosition={ControlPosition.TOP_LEFT}
                onPlaceSelect={setSelectedPlace}
              />
              <AutoCompleteResult place={selectedPlace} />
            </APIProvider>
          </Stack>
          <Grid container spacing={3}>
            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="roadClass" required>
                Road Class
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-roadClass"
                  id="select-roadClass"
                  value={roadClass}
                  label="Road Class"
                  onChange={setRoadClasshandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {roadClassRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="district" required>
                District
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-district"
                  id="select-district"
                  value={district}
                  label="District"
                  onChange={setDistricthandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {districtRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="collition" required>
                Collition Location
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-collition"
                  id="select-collition"
                  value={collitionLocation}
                  label="collition"
                  onChange={setcollitionLocationhandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {collitionLocationRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="traffic" required>
                Traffic
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-traffic"
                  id="select-traffic"
                  value={traffic}
                  label="traffic"
                  onChange={setTraffichandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {trafficRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="visibility" required>
                VISIBILITY
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-visibility"
                  id="select-visibility"
                  value={visibility}
                  label="visibility"
                  onChange={setVisibilityhandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {visibilityRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="light" required>
                Light
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-light"
                  id="select-light"
                  value={light}
                  label="light"
                  onChange={setLighthandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {lightRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="roadSurface" required>
                Road Surface
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-roadSurface"
                  id="select-roadSurface"
                  value={roadSurface}
                  label="roadSurface"
                  onChange={setRoadSurfaceLocationhandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {roadSurfaceRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="impactType" required>
                Impact Type
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-impactType"
                  id="select-impactType"
                  value={impactType}
                  label="impactType"
                  onChange={setImpactTypehandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {impactTypeRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="invType" required>
                Involvement Type
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-invType"
                  id="select-invType"
                  value={invtype}
                  label="invType"
                  onChange={setInvtypehandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {invtypeRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="age" required>
                Age
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-age"
                  id="select-age"
                  value={age}
                  label="age"
                  onChange={setAgehandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {ageRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="predestrianCond" required>
                Pedestrian Condition
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-predestrianCond"
                  id="select-predestrianCond"
                  value={pedestrian}
                  label="predestrianCond"
                  onChange={setPedestrianhandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {pedestrianRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="cyclistCond" required>
                Cyclist Condition
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-cyclistCond"
                  id="select-cyclistCond"
                  value={cyclist}
                  label="cyclistCond"
                  onChange={setCyclisthandleChange}
                  MenuProps={{
                    PaperProps: {
                      style: {
                        maxHeight: 200, // Adjust this value as needed
                      },
                    },
                  }}
                >
                  <MenuItem value="2023">
                  </MenuItem>
                  {cyclistRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormControlLabel
                control={<Checkbox name="isPedestrian" checked={isPedestrian} onChange={setisPedestrianhandleChange} value="yes" />}
                label="Pedestrian"
              />
              <FormControlLabel
                control={<Checkbox name="isCyclist" checked={isCyclist} onChange={setisCyclisthandleChange} value="yes" />}
                label="Cyclist"
              />
              <FormControlLabel
                control={<Checkbox name="isAutomobile" checked={isAutomobile} onChange={setisAutomobilehandleChange} value="yes" />}
                label="Automobile"
              />
              <FormControlLabel
                control={<Checkbox name="isMotorcycle" checked={isMotorcycle} onChange={setisMotorcyclehandleChange} value="yes" />}
                label="Motorcycle"
              />
              <FormControlLabel
                control={<Checkbox name="isSpeeding" checked={isSpeeding} onChange={setisSpeedinghandleChange} value="yes" />}
                label="Speeding"
              />
              <FormControlLabel
                control={<Checkbox name="isAggressive" checked={isAggressive} onChange={setisAggressivehandleChange} value="yes" />}
                label="Aggressive"
              />
              <FormControlLabel
                control={<Checkbox name="isRedLight" checked={isRedLight} onChange={setisRedLighthandleChange} value="yes" />}
                label="Red Light Related"
              />
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormControlLabel
                control={<Checkbox name="isTruck" checked={isTruck} onChange={setisTruckhandleChange} value="yes" />}
                label="Truck"
              />
              <FormControlLabel
                control={<Checkbox name="isCityVehicle" checked={isCityVehicle} onChange={setisCityVehiclehandleChange} value="yes" />}
                label="City Vehicle"
              />
              <FormControlLabel
                control={<Checkbox name="isEmergency" checked={isEmergency} onChange={setisEmergencyhandleChange} value="yes" />}
                label="Emergency Vehicle"
              />
              <FormControlLabel
                control={<Checkbox name="isPassenger" checked={isPassenger} onChange={setisPassengerhandleChange} value="yes" />}
                label="Passenger"
              />
              <FormControlLabel
                control={<Checkbox name="isAlcohol" checked={isAlcohol} onChange={setisAlcoholhandleChange} value="yes" />}
                label="Alcohol"
              />
              <FormControlLabel
                control={<Checkbox name="isDisability" checked={isDisability} onChange={setisDisabilityhandleChange} value="yes" />}
                label="Disability"
              />
            </FormGrid>

          </Grid>
        </Grid>

        <Box
          sx={[
            {
              display: 'flex',
              flexDirection: { xs: 'column-reverse', sm: 'row' },
              flexGrow: 1,
              gap: 1,
              pl: { xs: 4, sm: 4 },
              ml: { xs: 4, sm: 4 },
              mb: '60px'
            },
          ]}
        >

          <Grid size={{ xs: 5, sm: 5, lg: 5 }} direction='row' sx={{ justifyContent: 'center', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}>
            <Stack
              sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}
            >
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  height="250"
                  image={DT}
                  alt="DecisionTree"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Decision Tree
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    FATAL
                  </Typography>
                </CardContent>
              </Card>
            </Stack>
            <Stack
              sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}
            >
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  height="250"
                  image={RF}
                  alt="RandomForest"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Random Forest
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    FATAL
                  </Typography>
                </CardContent>
              </Card>
            </Stack>

            <Stack
              sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}
            >
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  height="250"
                  image={LR}
                  alt="Logistic"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Logistic Regression
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    FATAL
                  </Typography>
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          <Grid size={{ xs: 5, sm: 5, lg: 5 }} direction='row' sx={{ justifyContent: 'center', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}>
            <Stack
              sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}
            >
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  height="250"
                  image={SVM}
                  alt="SupportVector"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Support Vector
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    FATAL
                  </Typography>
                </CardContent>
              </Card>
            </Stack>
            <Stack
              sx={{ justifyContent: 'space-between', alignItems: 'center', padding: 1, 'padding-right': 1, 'padding-left': 1 }}
            >
              <Card sx={{ maxWidth: 345 }}>
                <CardMedia
                  component="img"
                  height="250"
                  image={NN}
                  alt="Neural Network"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Neural Network
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    FATAL
                  </Typography>
                </CardContent>
              </Card>
            </Stack>
          </Grid>
        </Box>
      </Grid>
    </Box >
  );
}
