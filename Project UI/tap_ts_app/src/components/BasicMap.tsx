import { useState, useEffect } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import FormLabel from '@mui/material/FormLabel';
import FormControl from '@mui/material/FormControl';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormHelperText from '@mui/material/FormHelperText';
import Typography from '@mui/material/Typography';
import {
  useMapsLibrary,
  AdvancedMarker,
  APIProvider,
  InfoWindow,
  Map,
  Marker,
  Pin
} from '@vis.gl/react-google-maps';

const API_KEY = "";

export default function BasicMap() {
  const apiUrl = 'http://127.0.0.1:5000'; 

  const maps = useMapsLibrary('maps'); // Access the core maps library
  const places = useMapsLibrary('places'); // Access the places library
  //const directions = useMapsLibrary('directions'); // Access the directions library

  type Poi = { key: string, location: google.maps.LatLngLiteral }
  const [fatalPoiList, setFatalPoiList] = useState<Poi[]>([]);
  const [nonPoiList, setNonPoiList] = useState<Poi[]>([]);
  const [propertyPoiList, setPropertyPoiList] = useState<Poi[]>([]);

  const [placesService, setPlacesService] = useState<google.maps.places.PlacesService | null>(null);

  const [state, setState] = useState({
    fatal: true,
    non: false,
    property: true,
  });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setState({
      ...state,
      [event.target.name]: event.target.checked,
    });
  };

  useEffect(() => {
    if (maps && places) {
      const mapElement = document.createElement('div'); // Create a dummy div for PlacesService
      setPlacesService(new places.PlacesService(mapElement));
    }
  }, [maps, places]);

  useEffect(() => {
        const fetchData = async () => {
  
          // Fetch the total accidents for the selected year for Fatalities
          try {
            const response = await fetch(apiUrl + '/data/map/basic', {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            });
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data: { 
              fatal: { key: string; location: google.maps.LatLngLiteral }[], 
              non: { key: string; location: google.maps.LatLngLiteral }[],
              property: { key: string; location: google.maps.LatLngLiteral }[]
            } = await response.json();
            setFatalPoiList(data.fatal.map(({ key, location }) => ({ key, location })));
            setNonPoiList(data.non.map(({ key, location }) => ({ key, location })));
            setPropertyPoiList(data.property.map(({ key, location }) => ({ key, location })));
            
  
          } catch (err) {
            console.error('Error fetching data:', err);
            console.error('Failed to load options. Please try again later.');
          } 
        };
    
        fetchData();
      }, []);

  const FatalPoiMarkers = (props: {pois: Poi[]}) => {
    return (
      <>
        {props.pois.map( (poi: Poi) => (
          <AdvancedMarker
            key={poi.key}
            position={poi.location}>
          <Pin background={'#CA2E55'} glyphColor={'#FFE0B5'} borderColor={'#000'} scale={0.4} />
          </AdvancedMarker>
        ))}
      </>
    );
  };

  const NonPoiMarkers = (props: {pois: Poi[]}) => {
    return (
      <>
        {props.pois.map( (poi: Poi) => (
          <AdvancedMarker
            key={poi.key}
            position={poi.location}>
          <Pin background={'#7A9E7E'} glyphColor={'#F2E7C9'} borderColor={'#000'} scale={0.4} />
          </AdvancedMarker>
        ))}
      </>
    );
  };

  const PropertyPoiMarkers = (props: {pois: Poi[]}) => {
    return (
      <>
        {props.pois.map( (poi: Poi) => (
          <AdvancedMarker
            key={poi.key}
            position={poi.location}>
          <Pin background={'#94A89A'} glyphColor={'#FFE0B5'} borderColor={'#000'} scale={0.4}/>
          </AdvancedMarker>
        ))}
      </>
    );
  };
  return (
    <Card variant="outlined" sx={{ width: '100%' }}>
      <CardContent>
      <Stack sx={{ justifyContent: 'space-between' }}>
            <Stack
              direction="row"
              sx={{ justifyContent: 'space-between', alignItems: 'center' }}
            >
              <Typography variant="h4" component="p">
                Pin Options
              </Typography>
            </Stack>
            <FormControl component="fieldset" variant="standard">
              <FormGroup aria-label="position" row>
                <FormControlLabel
                  control={
                    <Switch checked={state.fatal} onChange={handleChange} name="fatal" />
                  }
                  label="Fatal"
                />
                <FormControlLabel
                  control={
                    <Switch checked={state.non} onChange={handleChange} name="non" />
                  }
                  label="Non-Fatal"
                />
                <FormControlLabel
                  control={
                    <Switch checked={state.property} onChange={handleChange} name="property" />
                  }
                  label="Property Damage"
                />
              </FormGroup>
            </FormControl>
            
          </Stack>
        <div className="advanced-marker">
          <APIProvider apiKey={API_KEY} libraries={['marker']} solutionChannel='GMP_devsite_samples_v3_rgmbasicmap' onLoad={() => console.log('Maps API has loaded.')}>
            <Map
              mapId='Basic_Map'
              style={{width: '87vw', height: '75vh'}}
              defaultZoom={11}
              defaultCenter={{lat: 43.6532, lng: -79.3832}}
              gestureHandling={'greedy'}
              disableDefaultUI={true}
              >
              
              {state.non ? <NonPoiMarkers pois={nonPoiList} /> : null }
              {state.fatal ? <FatalPoiMarkers pois={fatalPoiList} /> : null }
              {state.property ? <PropertyPoiMarkers pois={propertyPoiList} /> : null }
            </Map>
          </APIProvider>
        </div>
      </CardContent>
    </Card>
  );
}
