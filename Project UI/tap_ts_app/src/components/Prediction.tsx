import { useState, useEffect, useCallback, ChangeEvent } from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Stack from '@mui/material/Stack';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CardMedia from '@mui/material/CardMedia';
import Avatar from '@mui/material/Avatar';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import dayjs, { Dayjs } from 'dayjs';

import DT from '../assets/decision-tree.png';
import RF from '../assets/forest.png';
import LR from '../assets/logistic-regression.png';
import SVM from '../assets/scatter-graph.png';
import NN from '../assets/deep-learning.png';
import P from '../assets/predictive.png';
import SV from '../assets/softvote.png';
import HV from '../assets/hardvote.png';

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

import weekOfYear from "dayjs/plugin/weekOfYear";

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

const API_KEY = "AIzaSyAPgyc_grB1usN6bQm5HEwSaBB8SK8lowg";
export type AutocompleteMode = { id: string; label: string };

export default function Prediction() {
  const apiUrl = 'http://127.0.0.1:5000';

  const maps = useMapsLibrary('maps'); // Access the core maps library
  const places = useMapsLibrary('places'); // Access the places library

  const [placesService, setPlacesService] = useState<google.maps.places.PlacesService | null>(null);

  const [svm, setsvm] = useState("");
  const [dt, setdt] = useState("");
  const [rf, setrf] = useState("");
  const [lr, setlr] = useState("");
  const [nn, setnn] = useState("");
  const [sv, setsv] = useState("");
  const [hv, sethv] = useState("");

  const [lat, setLat] = useState(0);
  const [long, setLong] = useState(0);

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

  const [isPedestrianM, setisPedestrianM] = useState("NO");
  const [isCyclistM, setisCyclistM] = useState("NO");
  const [isAutomobileM, setisAutomobileM] = useState("NO");
  const [isMotorcycleM, setisMotorcycleM] = useState("NO");
  const [isSpeedingM, setisSpeedingM] = useState("NO");
  const [isAggressiveM, setisAggressiveM] = useState("NO");
  const [isRedLightM, setisRedLightM] = useState("NO");
  const [isTruckM, setisTruckM] = useState("NO");
  const [isCityVehicleM, setisCityVehicleM] = useState("NO");
  const [isEmergencyM, setisEmergencyM] = useState("NO");
  const [isPassengerM, setisPassengerM] = useState("NO");
  const [isAlcoholM, setisAlcoholM] = useState("NO");
  const [isDisabilityM, setisDisabilityM] = useState("NO");

  const [roadClass, setRoadClass] = useState('LOCAL');
  const [district, setDistrict] = useState('NORTH YORK');
  const [collitionLocation, setcollitionLocation] = useState('AT INTERSECTION');
  const [traffic, setTraffic] = useState('NO CONTROL');
  const [visibility, setVisibility] = useState('CLEAR');
  const [light, setLight] = useState('DARK');
  const [roadSurface, setRoadSurface] = useState('DRY');
  const [impactType, setImpactType] = useState('APPROACHING');
  const [invtype, setInvtype] = useState('DRIVER');
  const [age, setAge] = useState('25 TO 29');
  const [pedestrian, setPedestrian] = useState('NORMAL');
  const [cyclist, setCyclist] = useState('NORMAL');
  const [neighbourhood, setNeighbourhood] = useState('WOODBINE-LUMSDEN');

  const [direction, setDirection] = useState('NORTH');
  const [vehicleType, setVehicleType] = useState('TAXI');
  const [manoeuver, setManoeuver] = useState('GOING AHEAD');
  const [driverAction, setDriverAction] = useState('DRIVING PROPERLY');
  const [driverCondition, setDriverCondition] = useState('NORMAL');
  const [cyclistAct, setCyclistAct] = useState('DRIVING PROPERLY');
  const [division, setDivision] = useState('D55');

  const [datetime, setDatetime] = useState<Dayjs | null>(dayjs());

  const [selectedPlace, setSelectedPlace] =
    useState<google.maps.places.Place | null>(null);

  const roadClassRef = ['MAJOR ARTERIAL', 'MINOR ARTERIAL', 'COLLECTOR', 'LOCAL', 'OTHER', 'PENDING', 'LANEWAY', 'EXPRESSWAY', 'EXPRESSWAY RAMP', 'MAJOR SHORELINE'];
  const districtRef = ['TORONTO AND EAST YORK', 'NORTH YORK', 'SCARBOROUGH', 'ETOBICOKE YORK', 'OTHER'];
  const collitionLocationRef = ['AT INTERSECTION', 'AT/NEAR PRIVATE DRIVE', 'INTERSECTION RELATED', 'LANEWAY', 'NON INTERSECTION', 'OTHER', 'OVERPASS OR BRIDGE', 'UNDERPASS OR TUNNEL'];
  const trafficRef = ['NO CONTROL', 'TRAFFIC SIGNAL', 'PEDESTRIAN CROSSOVER', 'STOP SIGN', 'OTHER', 'YIELD SIGN', 'TRAFFIC CONTROLLER', 'SCHOOL GUARD', 'POLICE CONTROL', 'TRAFFIC GATE', 'STREETCAR (STOP FOR)'];
  const visibilityRef = ['CLEAR', 'SNOW', 'OTHER', 'RAIN', 'STRONG WIND', 'FOG, MIST, SMOKE, DUST', 'DRIFTING SNOW', 'FREEZING RAIN'];
  const lightRef = ['DARK', 'DARK, ARTIFICIAL', 'DAYLIGHT', 'DUSK', 'DAWN', 'DUSK, ARTIFICIAL', 'DAWN, ARTIFICIAL', 'DAYLIGHT, ARTIFICIAL', 'OTHER'];
  const roadSurfaceRef = ['WET', 'SLUSH', 'DRY', 'ICE', 'LOOSE SNOW', 'OTHER', 'PACKED SNOW', 'SPILLED LIQUID', 'LOOSE SAND OR GRAVEL'];
  const directionRef = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'UNKNOWN'];
  const vehicleTypeRef = ['AUTOMOBILE, STATION WAGON', 'OTHER', 'PASSENGER VAN', 'MUNICIPAL TRANSIT BUS (TTC)', 'TAXI', 'BICYCLE', 'DELIVERY VAN', 'MOTORCYCLE', 'TRUCK - OPEN', 'MOPED', 'PICK UP TRUCK', 'TOW TRUCK', 'POLICE VEHICLE', 'TRUCK-TRACTOR', 'STREET CAR', 'TRUCK - CLOSED (BLAZER, ETC)', 'TRUCK - DUMP', 'BUS (OTHER) (GO BUS, GRAY COA', 'CONSTRUCTION EQUIPMENT', 'INTERCITY BUS', 'TRUCK (OTHER)', 'FIRE VEHICLE', 'SCHOOL BUS', 'OTHER EMERGENCY VEHICLE', 'OFF ROAD - 2 WHEELS', 'TRUCK - TANK', 'TRUCK - CAR CARRIER', 'AMBULANCE', 'OFF ROAD - 4 WHEELS', 'OFF ROAD - OTHER', 'RICKSHAW', 'UNKNOWN'];
  const manoeuverRef = ['GOING AHEAD', 'CHANGING LANES', 'TURNING RIGHT', 'SLOWING OR STOPPING', 'TURNING LEFT', 'OTHER', 'STOPPED', 'UNKNOWN', 'PARKED', 'OVERTAKING', 'MAKING U TURN', 'REVERSING', 'PULLING AWAY FROM SHOULDER OR CURB', 'PULLING ONTO SHOULDER OR TOWARDCURB', 'MERGING', 'DISABLED'];
  const driverActionRef = ['DRIVING PROPERLY', 'LOST CONTROL', 'IMPROPER LANE CHANGE', 'DISOBEYED TRAFFIC CONTROL', 'FAILED TO YIELD RIGHT OF WAY', 'OTHER', 'SPEED TOO FAST FOR CONDITION', 'EXCEEDING SPEED LIMIT', 'IMPROPER TURN', 'FOLLOWING TOO CLOSE', 'IMPROPER PASSING', 'WRONG WAY ON ONE WAY ROAD', 'SPEED TOO SLOW'];
  const driverConditionRef = ['NORMAL', 'ABILITY IMPAIRED, ALCOHOL OVER .08', 'INATTENTIVE', 'UNKNOWN', 'MEDICAL OR PHYSICAL DISABILITY', 'HAD BEEN DRINKING', 'FATIGUE', 'OTHER', "ABILITY IMPAIRED, ALCOHOL", "ABILITY IMPAIRED, DRUGS"];
  const cyclistActRef = ['DRIVING PROPERLY', 'OTHER', 'IMPROPER TURN', 'IMPROPER PASSING', 'DISOBEYED TRAFFIC CONTROL', 'LOST CONTROL', 'FAILED TO YIELD RIGHT OF WAY', 'IMPROPER LANE CHANGE', 'FOLLOWING TOO CLOSE', 'SPEED TOO FAST FOR CONDITION', 'WRONG WAY ON ONE WAY ROAD']
  const divisionRef = ["D55","D14","D11","D33","D42","D51","D23","D41","D31","D53","D32","D43","D22","D13","D52","D12","NSA"]
  const impactTypeRef = ['APPROACHING', 'SMV OTHER', 'PEDESTRIAN COLLISIONS', 'ANGLE', 'TURNING MOVEMENT', 'CYCLIST COLLISIONS', 'REAR END', 'SIDESWIPE', 'SMV UNATTENDED VEHICLE', 'OTHER'];
  const invtypeRef = ['PASSENGER', 'DRIVER', 'VEHICLE OWNER', 'OTHER PROPERTY OWNER', 'PEDESTRIAN', 'CYCLIST', 'OTHER', 'MOTORCYCLE DRIVER', 'TRUCK DRIVER', 'IN-LINE SKATER', 'DRIVER - NOT HIT', 'MOTORCYCLE PASSENGER', 'MOPED DRIVER', 'WHEELCHAIR', 'PEDESTRIAN - NOT HIT', 'TRAILER OWNER', 'WITNESS', 'CYCLIST PASSENGER', 'MOPED PASSENGER'];
  const ageRef = ['UNKNOWN', '0 TO 4', '5 TO 9', '10 TO 14', '15 TO 19', '20 TO 24', '25 TO 29', '30 TO 34', '35 TO 39', '40 TO 44', '45 TO 49', '50 TO 54', '55 TO 59', '60 TO 64', '65 TO 69', '70 TO 74', '75 TO 79', '80 TO 84', '85 TO 89', '90 TO 94', 'OVER 95'];
  const pedestrianRef = ['NA', 'INATTENTIVE', 'NORMAL', 'UNKNOWN', 'MEDICAL OR PHYSICAL DISABILITY', 'HAD BEEN DRINKING', 'ABILITY IMPAIRED, ALCOHOL', 'OTHER', 'ABILITY IMPAIRED, ALCOHOL OVER .80', 'ABILITY IMPAIRED, DRUGS', 'FATIGUE'];
  const cyclistRef = ['NA', 'NORMAL', 'INATTENTIVE', 'HAD BEEN DRINKING', 'UNKNOWN', 'ABILITY IMPAIRED, DRUGS', 'ABILITY IMPAIRED, ALCOHOL OVER .80', 'MEDICAL OR PHYSICAL DISABILITY', 'ABILITY IMPAIRED, ALCOHOL', 'OTHER', 'FATIGUE'];
  const neighbourhoodRef = ['WOODBINE-LUMSDEN', 'WOODBINE CORRIDOR', 'KENSINGTON-CHINATOWN', 'DUFFERIN GROVE', 'DON VALLEY VILLAGE', 'MORNINGSIDE HEIGHTS', 'ST LAWRENCE-EAST BAYFRONT-THE ISLANDS', 'ELMS-OLD REXDALE', 'DORSET PARK', 'AGINCOURT NORTH', 'BENDALE SOUTH',
    'VICTORIA VILLAGE', 'HUMBERMEDE', 'YONGE-EGLINTON', 'RUNNYMEDE-BLOOR WEST VILLAGE', 'LANSING-WESTGATE', 'WEST HILL', 'AGINCOURT SOUTH-MALVERN WEST', 'ANNEX', 'WEXFORD/MARYVALE', 'WEST ROUGE', 'ROSEDALE-MOORE PARK', 'PALMERSTON-LITTLE ITALY', 'MIMICO-QUEENSWAY',
    'CASA LOMA', "EAST L'AMOREAUX", 'HIGH PARK NORTH', 'WEST HUMBER-CLAIRVILLE', "PARKWOODS-O'CONNOR HILLS", 'IONVIEW', 'DANFORTH', "O'CONNOR-PARKVIEW", 'KEELESDALE-EGLINTON WEST', 'DANFORTH EAST YORK', 'REXDALE-KIPLING', 'DOVERCOURT VILLAGE', 'LEASIDE-BENNINGTON',
    'SOUTH PARKDALE', 'MALVERN WEST', 'ETOBICOKE CITY CENTRE', 'FOREST HILL SOUTH', 'ERINGATE-CENTENNIAL-WEST DEANE', 'MOSS PARK', 'SOUTH RIVERDALE', 'EGLINTON EAST', 'BROADVIEW NORTH', 'HIGH PARK-SWANSEA', 'HUMBER BAY SHORES', 'MALVERN EAST', 'KENNEDY PARK',
    'TRINITY-BELLWOODS', 'KINGSVIEW VILLAGE-THE WESTWAY', 'STEELES', 'JUNCTION-WALLACE EMERSON', 'EAST WILLOWDALE', 'YORK UNIVERSITY HEIGHTS', 'WESTON-PELHAM PARK', 'NSA', 'WOBURN NORTH', 'BROOKHAVEN-AMESBURY', 'MAPLE LEAF', 'ROCKCLIFFE-SMYTHE', 'CORSO ITALIA-DAVENPORT',
    'MOUNT DENNIS', 'HARBOURFRONT-CITYPLACE', 'FLEMINGDON PARK', 'BANBURY-DON MILLS', 'YONGE-BAY CORRIDOR', 'ENGLEMOUNT-LAWRENCE', 'SCARBOROUGH VILLAGE', 'BAY-CLOVERHILL', 'GLENFIELD-JANE HEIGHTS', 'CLAIRLEA-BIRCHMOUNT', 'LAWRENCE PARK SOUTH', 'BEDFORD PARK-NORTOWN',
    'BIRCHCLIFFE-CLIFFSIDE', 'FOREST HILL NORTH', 'HUMBER SUMMIT', 'BEECHBOROUGH-GREENBROOK', 'WILLOWDALE WEST', 'GREENWOOD-COXWELL', 'ST.ANDREW-WINDFIELDS', 'EDENBRIDGE-HUMBER VALLEY', 'MOUNT PLEASANT EAST', 'STONEGATE-QUEENSWAY', 'YONGE-ST.CLAIR', 'HUMEWOOD-CEDARVALE',
    'OAKDALE-BEVERLEY HEIGHTS', 'WESTMINSTER-BRANSON', 'HENRY FARM', 'DOWNTOWN YONGE EAST', 'BLACK CREEK', 'HUMBER HEIGHTS-WESTMOUNT', 'CABBAGETOWN-SOUTH ST.JAMES TOWN', 'THE BEACHES', 'WYCHWOOD', 'MORNINGSIDE', 'SOUTH EGLINTON-DAVISVILLE', 'WEST QUEEN WEST',
    'YONGE-DORIS', 'NEW TORONTO', 'CLANTON PARK', 'PRINCESS-ROSETHORN', 'FENSIDE-PARKWOODS', 'OAKWOOD VILLAGE', 'BENDALE-GLEN ANDREW', 'WELLINGTON PLACE', 'LITTLE PORTUGAL', 'LAMBTON BABY POINT', 'OLD EAST YORK', 'HILLCREST VILLAGE', 'AVONDALE', 'ALDERWOOD',
    'TAYLOR-MASSEY', 'NEWTONBROOK EAST', 'CLIFFCREST', 'CALEDONIA-FAIRBANK', 'CHURCH-WELLESLEY', 'MILLIKEN', 'BATHURST MANOR', 'BRIAR HILL-BELGRAVIA', 'FORT YORK-LIBERTY VILLAGE', 'BAYVIEW VILLAGE', 'PELMO PARK-HUMBERLEA', 'THORNCLIFFE PARK', 'ETOBICOKE WEST MALL',
    'WESTON', 'WILLOWRIDGE-MARTINGROVE-RICHVIEW', 'YORKDALE-GLEN PARK', 'NORTH RIVERDALE', 'RONCESVALLES', 'MOUNT OLIVE-SILVERSTONE-JAMESTOWN', "TAM O'SHANTER-SULLIVAN", 'THISTLETOWN-BEAUMOND HEIGHTS', "L'AMOREAUX WEST", 'REGENT PARK', 'DOWNSVIEW',
    'BRIDLE PATH-SUNNYBROOK-YORK MILLS', 'GUILDWOOD', 'ISLINGTON', 'LAWRENCE PARK NORTH', 'RUSTIC', 'PLEASANT VIEW', 'GOLFDALE-CEDARBRAE-WOBURN', 'EAST END-DANFORTH', 'JUNCTION AREA', 'UNIVERSITY', 'NEWTONBROOK WEST', 'HIGHLAND CREEK', 'CENTENNIAL SCARBOROUGH',
    'BLAKE-JONES', 'OAKRIDGE', 'BAYVIEW WOODS-STEELES', 'LONG BRANCH', 'KINGSWAY SOUTH', 'MARKLAND WOOD', 'PLAYTER ESTATES-DANFORTH', 'NORTH ST.JAMES TOWN', 'NORTH TORONTO'];

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
  const setNeighbourhoodhandleChange = (event: SelectChangeEvent) => {
    setNeighbourhood(event.target.value);
  };
  const setDirectionhandleChange = (event: SelectChangeEvent) => {
    setDirection(event.target.value);
  };
  const setVehicleTypehandleChange = (event: SelectChangeEvent) => {
    setVehicleType(event.target.value);
  };
  const setManoeuverhandleChange = (event: SelectChangeEvent) => {
    setManoeuver(event.target.value);
  };
  const setDriverActionhandleChange = (event: SelectChangeEvent) => {
    setDriverAction(event.target.value);
  };
  const setDriverConditionhandleChange = (event: SelectChangeEvent) => {
    setDriverCondition(event.target.value);
  };
  const setCyclistActhandleChange = (event: SelectChangeEvent) => {
    setCyclistAct(event.target.value);
  };
  const setDivisionhandleChange = (event: SelectChangeEvent) => {
    setDivision(event.target.value);
  };

  const setisPedestrianhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisPedestrian(event.target.checked);
    if (event.target.checked) {
      setisPedestrianM("YES");
    } else {
      setisPedestrianM("NO");
    }
  };
  const setisCyclisthandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisCyclist(event.target.checked);
    if (event.target.checked) {
      setisCyclistM("YES");
    } else {
      setisCyclistM("NO");
    }
  };
  const setisAutomobilehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAutomobile(event.target.checked);
    if (event.target.checked) {
      setisAutomobileM("YES");
    } else {
      setisAutomobileM("NO");
    }
  };
  const setisMotorcyclehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisMotorcycle(event.target.checked);
    if (event.target.checked) {
      setisMotorcycleM("YES");
    } else {
      setisMotorcycleM("NO");
    }
  };
  const setisSpeedinghandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisSpeeding(event.target.checked);
    if (event.target.checked) {
      setisSpeedingM("YES");
    } else {
      setisSpeedingM("NO");
    }
  };
  const setisAggressivehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAggressive(event.target.checked);
    if (event.target.checked) {
      setisAggressiveM("YES");
    } else {
      setisAggressiveM("NO");
    }
  };
  const setisRedLighthandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisRedLight(event.target.checked);
    if (event.target.checked) {
      setisRedLightM("YES");
    } else {
      setisRedLightM("NO");
    }
  };
  const setisTruckhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisTruck(event.target.checked);
    if (event.target.checked) {
      setisTruckM("YES");
    } else {
      setisTruckM("NO");
    }
  };
  const setisCityVehiclehandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisCityVehicle(event.target.checked);
    if (event.target.checked) {
      setisCityVehicleM("YES");
    } else {
      setisCityVehicleM("NO");
    }
  };
  const setisEmergencyhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisEmergency(event.target.checked);
    if (event.target.checked) {
      setisEmergencyM("YES");
    } else {
      setisEmergencyM("NO");
    }
  };
  const setisPassengerhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisPassenger(event.target.checked);
    if (event.target.checked) {
      setisPassengerM("YES");
    } else {
      setisPassengerM("NO");
    }
  };
  const setisAlcoholhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisAlcohol(event.target.checked);
    if (event.target.checked) {
      setisAlcoholM("YES");
    } else {
      setisAlcoholM("NO");
    }
  };
  const setisDisabilityhandleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setisDisability(event.target.checked);
    if (event.target.checked) {
      setisDisabilityM("YES");
    } else {
      setisDisabilityM("NO");
    }
  };

  const mapClick = useCallback((ev: MapMouseEvent) => {
    if (ev.detail.latLng) {
      console.log('marker clicked:', ev.detail.latLng);
      setLat(ev.detail.latLng.lat);
      setLong(ev.detail.latLng.lng);

    } else {
      console.log('marker clicked: latLng is null');
    }
  }, []);

  const handleLongitudeChange = (event: ChangeEvent<HTMLInputElement>) => {
    setLong(Number (event.target.value));
  };
  const handleLatitudeChange = (event: ChangeEvent<HTMLInputElement>) => {
    setLat(Number (event.target.value));
  };

  const handlePrediction = async () => {
    console.log("PREDICTION")

    dayjs.extend(weekOfYear)

    // Form the data object
    const new_data = {
      'ROAD_CLASS': roadClass.toUpperCase(),
      'DISTRICT': district.toUpperCase(),
      'LATITUDE': lat,
      'LONGITUDE': long,
      'ACCLOC': collitionLocation.toUpperCase(),
      'TRAFFCTL': traffic.toUpperCase(),
      'VISIBILITY': visibility.toUpperCase(),
      'LIGHT': light.toUpperCase(),
      'RDSFCOND': roadSurface.toUpperCase(),
      'IMPACTYPE': impactType.toUpperCase(),
      'INVTYPE': invtype.toUpperCase(),
      'INVAGE': age.toUpperCase(),
      'PEDCOND': pedestrian.toUpperCase(),
      'CYCCOND': cyclist.toUpperCase(),
      'PEDESTRIAN': isPedestrianM,
      'CYCLIST': isCyclistM,
      'AUTOMOBILE': isAutomobileM,
      'MOTORCYCLE': isMotorcycleM,
      'TRUCK': isTruckM,
      'TRSN_CITY_VEH': isCityVehicleM,
      'EMERG_VEH': isEmergencyM,
      'PASSENGER': isPassengerM,
      'SPEEDING': isSpeedingM,
      'AG_DRIV': isAggressiveM,
      'REDLIGHT': isRedLightM,
      'ALCOHOL': isAlcoholM,
      'DISABILITY': isDisabilityM,
      'NEIGHBOURHOOD_158': neighbourhood.toUpperCase(),
      'HOUR': datetime?.hour(),
      'MONTH': (datetime?.month() ?? -1) + 1,
      'DAY': datetime?.date(),
      'WEEK': datetime?.week(),
      'DAYOFWEEK': (datetime?.day() ?? -1) + 1 == 7 ? 0 : (datetime?.day() ?? -1) + 1,
      'ACCLASS': 'Fatal'
    }
    // Neural Network
    try {
      const response = await fetch(apiUrl + '/predict/nn', {method: 'POST', body: JSON.stringify(new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setnn(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }

    // Decision Tree
    try {
      const response = await fetch(apiUrl + '/predict/dt', {method: 'POST', body: JSON.stringify(new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setdt(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }

    // Random Forest
    try {
      const response = await fetch(apiUrl + '/predict/rf', {method: 'POST', body: JSON.stringify(new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setrf(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }

    // SVM
    const svm_new_data = {
      'ROAD_CLASS': roadClass.toUpperCase(),
      'DISTRICT': district.toUpperCase(),
      'LATITUDE': lat,
      'LONGITUDE': long,
      'ACCLOC': collitionLocation.toUpperCase(),
      'TRAFFCTL': traffic.toUpperCase(),
      'VISIBILITY': visibility.toUpperCase(),
      'LIGHT': light.toUpperCase(),
      'RDSFCOND': roadSurface.toUpperCase(),
      'IMPACTYPE': impactType.toUpperCase(),
      'INVTYPE': invtype.toUpperCase(),
      'INVAGE': age.toUpperCase(),
      'PEDCOND': pedestrian.toUpperCase(),
      'CYCCOND': cyclist.toUpperCase(),
      'PEDESTRIAN': isPedestrianM,
      'CYCLIST': isCyclistM,
      'AUTOMOBILE': isAutomobileM,
      'MOTORCYCLE': isMotorcycleM,
      'TRUCK': isTruckM,
      'TRSN_CITY_VEH': isCityVehicleM,
      'EMERG_VEH': isEmergencyM,
      'PASSENGER': isPassengerM,
      'SPEEDING': isSpeedingM,
      'AG_DRIV': isAggressiveM,
      'REDLIGHT': isRedLightM,
      'ALCOHOL': isAlcoholM,
      'DISABILITY': isDisabilityM,
      'NEIGHBOURHOOD_158': neighbourhood.toUpperCase(),
      'MONTH': (datetime?.month() ?? -1) + 1,
      'DAY': datetime?.date(),
      'WEEK': datetime?.week(),
      'DAYOFWEEK': (datetime?.day() ?? -1) + 1 == 7 ? 0 : (datetime?.day() ?? -1) + 1,
      'SEASON': 0,
      'HOUR': datetime?.hour(),
      'MINUTE': datetime?.minute(),
    }

    try {
      const response = await fetch(apiUrl + '/predict/svm', {method: 'POST', body: JSON.stringify(svm_new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setsvm(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }

    // Softvoting and Hard voting
    const voting_new_data = {
      'ROAD_CLASS': roadClass.toUpperCase(),
      'DISTRICT': district.toUpperCase(),
      'LATITUDE': lat,
      'LONGITUDE': long,
      'ACCLOC': collitionLocation.toUpperCase(),
      'TRAFFCTL': traffic.toUpperCase(),
      'VISIBILITY': visibility.toUpperCase(),
      'LIGHT': light.toUpperCase(),
      'RDSFCOND': roadSurface.toUpperCase(),
      'IMPACTYPE': impactType.toUpperCase(),
      'INVTYPE': invtype.toUpperCase(),
      'INITDIR': direction.toUpperCase(),
      'INVAGE': age.toUpperCase(),
      'VEHTYPE': vehicleType.toUpperCase(),
      'MANOEUVER': manoeuver.toUpperCase(),
      'DRIVACT': driverAction.toUpperCase(),
      'DRIVCOND': driverCondition.toUpperCase(),
      'PEDCOND': pedestrian.toUpperCase(),
      'CYCACT': cyclistAct.toUpperCase(),
      'PEDESTRIAN': isPedestrianM,
      'CYCLIST': isCyclistM,
      'AUTOMOBILE': isAutomobileM,
      'MOTORCYCLE': isMotorcycleM,
      'TRUCK': isTruckM,
      'TRSN_CITY_VEH': isCityVehicleM,
      'PASSENGER': isPassengerM,
      'SPEEDING': isSpeedingM,
      'AG_DRIV': isAggressiveM,
      'REDLIGHT': isRedLightM,
      'ALCOHOL': isAlcoholM,
      'DISABILITY': isDisabilityM,
      'NEIGHBOURHOOD_158': neighbourhood.toUpperCase(),
      'HOOD_158': '',
      'HOOD_140': '',
      'DIVISION': division.toUpperCase(),
      'MONTH': (datetime?.month() ?? -1) + 1,
      'SEASON': 0,
      'HOUR': datetime?.hour(),
      'MINUTE': datetime?.minute(),
    }

    try {
      const response = await fetch(apiUrl + '/predict/sv', {method: 'POST', body: JSON.stringify(voting_new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setsvm(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }

    try {
      const response = await fetch(apiUrl + '/predict/hv', {method: 'POST', body: JSON.stringify(voting_new_data), headers: {'Content-Type': 'application/json'}});
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setsvm(data["prediction"]);

    } catch (err) {
      console.error('Error fetching data:', err);
      console.error('Failed to load options. Please try again later.');
    }
  };
  useEffect(() => {
    if (maps && places) {
      const mapElement = document.createElement('div'); // Create a dummy div for PlacesService
      setPlacesService(new places.PlacesService(mapElement));
    }
  }, [maps, places]);


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
              <LocalizationProvider dateAdapter={AdapterDayjs}>
              <DateTimePicker
                label="Date and Time"
                value={datetime}
                onChange={(newValue) => setDatetime(newValue)}
              />
              </LocalizationProvider>
            </CardContent>
          </Card>

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
                    onChange={handleLatitudeChange}
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
                    onChange={handleLongitudeChange}
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
              <FormLabel htmlFor="neighbourhood" required>
                NEIGHBOURHOOD
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-neighbourhood"
                  id="select-neighbourhood"
                  value={neighbourhood}
                  label="Neighbourhood"
                  onChange={setNeighbourhoodhandleChange}
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
                  {neighbourhoodRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

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
              <FormLabel htmlFor="direction">
                Direction
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-direction"
                  id="select-direction"
                  value={direction}
                  label="involvement"
                  onChange={setDirectionhandleChange}
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
                  {directionRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="vehicleType">
                Vehicle Type
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-vehicleType"
                  id="select-vehicleType"
                  value={vehicleType}
                  label="vehicleType"
                  onChange={setVehicleTypehandleChange}
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
                  {vehicleTypeRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="manoeuver">
                Vehicle Manoeuver
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-manoeuver"
                  id="select-manoeuver"
                  value={manoeuver}
                  label="vehicleType"
                  onChange={setManoeuverhandleChange}
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
                  {manoeuverRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="driverAction">
                Driver Action
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-driverAction"
                  id="select-driverAction"
                  value={driverAction}
                  label="vehicleType"
                  onChange={setDriverActionhandleChange}
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
                  {driverActionRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="driverCondition">
                Driver Condition
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-driverCondition"
                  id="select-driverCondition"
                  value={driverCondition}
                  label="vehicleType"
                  onChange={setDriverConditionhandleChange}
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
                  {driverConditionRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="cyclistAct">
                Cyclist Action
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-cyclistAct"
                  id="select-cyclistAct"
                  value={cyclistAct}
                  label="vehicleType"
                  onChange={setCyclistActhandleChange}
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
                  {cyclistActRef.map((option) => (
                    <MenuItem key={option} value={option}>
                      {String(option)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </FormGrid>

            <FormGrid size={{ xs: 12, md: 6 }}>
              <FormLabel htmlFor="division">
                Division
              </FormLabel>
              <FormControl sx={{ m: 0, minWidth: 120 }}>
                <Select
                  labelId="select-division"
                  id="select-division"
                  value={division}
                  label="division"
                  onChange={setDivisionhandleChange}
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
                  {divisionRef.map((option) => (
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
                    {dt}
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
                    {rf}
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
                    {lr}
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
                  image={HV}
                  alt="Hard Voting"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Hard Voting
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    {hv}
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
                    {svm}
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
                    {nn}
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
                  image={SV}
                  alt="Soft Voting"
                />
                <CardContent>
                  <Typography gutterBottom variant="h5" component="div">
                    Soft Voting
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    {sv}
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
