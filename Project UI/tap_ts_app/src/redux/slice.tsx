import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { PURGE } from "redux-persist";

// Define the state interface
interface AppState {
  year: string;
}

// Initial state
const initialState: AppState = {
  year: '2023'
};

// Create a slice
const appSlice = createSlice({
  name: "tapApp",
  initialState,
  reducers: {
    setYear: (state, action: PayloadAction<string>) => {
      state.year = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(PURGE, () => {
      return initialState;
    });
  }
});

// Export actions
export const { setYear } = appSlice.actions;

// Export the reducer
export default appSlice.reducer;