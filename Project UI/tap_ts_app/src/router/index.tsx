import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from '../Dashboard';

function AppRouter() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/home" element={<Dashboard />} />
                {/* Add more routes here as needed */}
            </Routes>
        </BrowserRouter>
    );
}

export default AppRouter;
