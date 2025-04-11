import { configureStore } from "@reduxjs/toolkit";
import tapApp from "./slice";
import {
  persistStore,
  persistReducer,
  FLUSH,
  REHYDRATE,
  PAUSE,
  PERSIST,
  PURGE,
  REGISTER,
} from "redux-persist";
import storage from "redux-persist/lib/storage";

// Define the persist configuration
const persistConfig = {
  key: "root",
  storage,
};

// Create a persisted reducer
const persistedReducer = persistReducer(persistConfig, tapApp);

// Configure the Redux store
export const tapStore = configureStore({
  reducer: {
    tapApp: persistedReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [FLUSH, REHYDRATE, PAUSE, PERSIST, PURGE, REGISTER],
      },
    }),
});

// Create a persistor
export const persistor = persistStore(tapStore);

// Define the RootState and AppDispatch types
export type RootState = ReturnType<typeof tapStore.getState>;
export type AppDispatch = typeof tapStore.dispatch;
