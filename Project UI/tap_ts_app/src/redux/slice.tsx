import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { PURGE } from "redux-persist";

// Define the state interface
interface AppState {
  year: string;
  model: string;
}

// Initial state
const initialState: AppState = {
  year: '2023',
  model: 'lg',
};

// Create a slice
const appSlice = createSlice({
  name: "tapApp",
  initialState,
  reducers: {
    setYear: (state, action: PayloadAction<string>) => {
      state.year = action.payload;
    },
    setModel: (state, action: PayloadAction<string>) => {
      state.model = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(PURGE, () => {
      return initialState;
    });
  }
});

// Export actions
export const { setYear, setModel } = appSlice.actions;

// Export the reducer
export default appSlice.reducer;