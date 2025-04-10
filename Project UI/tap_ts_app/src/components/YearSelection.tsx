import * as React from 'react';
import { useState, useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../redux/store";
import { AppDispatch } from '../redux/store';
import { setYear } from "../redux/slice";
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';

export default function YearSelection() {
    const year = useSelector((state: RootState) => state.tapApp.year);
    const [options, setOptions] = useState<ApiResponseItem[]>([]);
    const apiUrl = 'http://127.0.0.1:5000'; // Replace with your actual API endpoint
    
    const dispatch = useDispatch<AppDispatch>();

    const handleChange = (event: SelectChangeEvent) => {
        setYear(event.target.value);
        dispatch(setYear(String(event.target.value)));
    };
    
    interface ApiResponseItem {
        value: number;
        label: number;
      }

    useEffect(() => {
        const fetchData = async () => {

          try {
            const response = await fetch(apiUrl + '/data/year_list');
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data: number[] = await response.json();
            setOptions(data.map(item => ({ value: item, label: item })));
          } catch (err) {
            console.error('Error fetching data:', err);
            console.error('Failed to load options. Please try again later.');
          } 
        };
    
        fetchData();
      }, [apiUrl]);

    return (
      <div>
        <FormControl sx={{ m: 0, minWidth: 120}}>
          <Select
            labelId="select-year"
            id="select-year"
            value={year}
            label="Year"
            onChange={handleChange}
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
            {options.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                    {String(option.label)}
                </MenuItem>
            ))}
          </Select>
        </FormControl>
      </div>
    );
  }